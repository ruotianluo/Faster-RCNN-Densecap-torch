# coding=utf8

import argparse, os, json, string
from collections import Counter
from Queue import Queue
from threading import Thread, Lock

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize
import os

import xmltodict

"""
This file expects a VOC style data.
0000xx.xml contains:
folder:
filename:
source:
owner:
size:
segmented:
object:
    name: class label
    pose:
    truncated: 0,1
    difficult: 0,1
    bndbox:
        xmin
        ymin
        xmax
        ymax

The top-left pixel in the image has coordinate (1,1)

The data will be preprocessed into an HDF5 file and a JSON file with
some auxiliary information. The class label will be indexed by a number.

Note, in general any indices anywhere in input/output of this file are 1-indexed.

The output JSON file is an object with the following elements:
- cls_to_idx: class labels to integers, 
                in 1-indexed format.
- filename_to_idx: Dictionary mapping string filenames to indices.
- idx_to_cls: Inverse of the above.
- idx_to_filename: Inverse of the above.

The output HDF5 file has the following format to describe N images with
M total regions:

- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (x, y, w, h) where y and x are top-left coordinates of the box,
  and are one-indexed.
- labels: int32 array of shape (M,) giving the class label index  for each region.
  To recover a class label from an integer in this matrix,
  use idx_to_cls from the JSON output file.
- difficult: int32 array of shape(M, ) giving if a region is difficult or not
- img_to_first_box: int32 array of shape (N,). If img_to_first_box[i] = j then
  labels[j] and boxes[j] give the first annotation for image i
  (using one-indexing).
- img_to_last_box: int32 array of shape (N,). If img_to_last_box[i] = j then
  labels[j] and boxes[j] give the last annotation for image i
  (using one-indexing).
- box_to_img: int32 array of shape (M,). If box_to_img[i] = j then then
  regions[i] and captions[i] refer to images[j] (using one-indexing).
"""

def build_cls_dict():
  classes = ['__background__', 'aeroplane','bicycle','bird','boat','bottle','bus','car',
              'cat','chair','cow','diningtable','dog','horse','motorbike',
              'person','pottedplant','sheep','sofa','train','tvmonitor']
  cls_to_idx, idx_to_cls = {}, {}
  next_idx = 1

  for cls in classes:
    cls_to_idx[cls] = next_idx
    idx_to_cls[next_idx] = cls
    next_idx = next_idx + 1
    
  return cls_to_idx, idx_to_cls

def getAnnotations(annopath, img_id):
  anno = xmltodict.parse(open(annopath %(img_id)))
  return anno['annotation']

def getAllAnnotations(annopath, imgpath, all_data):
  data = []
  for img_id in all_data:
    tmp = getAnnotations(annopath, img_id)
    tmp['id'] = img_id
    tmp['filename'] = imgpath %(img_id)
    if not type(tmp['object']) is list:
      tmp['object'] = [tmp['object']]
    tmp_list = []
    for obj in tmp['object']:
      if int(obj['difficult']) == 0:
        tmp_list.append(obj)
    tmp['object'] = tmp_list
    if len(tmp_list) > 0 and os.path.isfile(tmp['filename']):
      data.append(tmp)
    else:
      print 'no objects or no corresponding image'
  return data

def encode_labels(data, cls_to_idx):
  encoded_list = []
  difficult = []
  for i, img in enumerate(data):
    for region in img['object']:
      encoded_list.append(cls_to_idx[region['name']])
      difficult.append(region['difficult'])
  return np.asarray(encoded_list, dtype=np.int32), np.asarray(difficult, dtype=np.int32)

def encode_boxes(data):
  all_boxes = []
  for i, img in enumerate(data):
    for region in img['object']:
      if region['name'] is None: continue
      # recall: x,y are 1-indexed
      x, y = int(region['bndbox']['xmin']), int(region['bndbox']['ymin'])
      w = int(region['bndbox']['xmax'])-int(region['bndbox']['xmin'])
      h = int(region['bndbox']['ymax'])-int(region['bndbox']['ymin'])

      box = np.asarray([x, y, w, h], dtype=np.int32) # also convert to center-coord oriented
      assert box[2]>=0 # width height should be positive numbers
      assert box[3]>=0
      all_boxes.append(box)

  return np.vstack(all_boxes)

def build_img_idx_to_box_idxs(data):
  img_idx = 1
  box_idx = 1
  num_images = len(data)
  img_to_first_box = np.zeros(num_images, dtype=np.int32)
  img_to_last_box = np.zeros(num_images, dtype=np.int32)
  for img in data:
    img_to_first_box[img_idx - 1] = box_idx
    for region in img['object']:
      if region['name'] is None: continue
      box_idx += 1
    img_to_last_box[img_idx - 1] = box_idx - 1 # -1 to make these inclusive limits
    img_idx += 1
  
  return img_to_first_box, img_to_last_box

def build_filename_dict(all_data, imgpath):
  # First make sure all filenames
  assert len(all_data) == len(set(all_data))
  
  next_idx = 1
  filename_to_idx, idx_to_filename = {}, {}
  for img_id in all_data:
    filename = imgpath %(img_id)
    filename_to_idx[filename] = next_idx
    idx_to_filename[next_idx] = filename
    next_idx += 1
  return filename_to_idx, idx_to_filename

def encode_filenames(data, imgpath, filename_to_idx):
  filename_idxs = []
  for img in data:
    filename = imgpath %(img['id'])
    idx = filename_to_idx[filename]
    for region in img['object']:
      if region['name'] is None: continue
      filename_idxs.append(idx)
  return np.asarray(filename_idxs, dtype=np.int32)

def encode_splits(data, split_data):
  """ Encode splits as intetgers and return the array. """
  lookup = {'train': 0, 'val': 1, 'test': 2}
  id_to_split = {}
  split_array = np.zeros(len(data))
  for split, idxs in split_data.iteritems():
    for idx in idxs:
      id_to_split[idx] = split
  for i, img_id in enumerate(data):
    split_array[i] = lookup[id_to_split[img_id]]
  return split_array


def filter_images(data, split_data):
  """ Keep only images that are in some split and have some captions """
  all_split_ids = set()
  for split_name, ids in split_data.iteritems():
    all_split_ids.update(ids)
  new_data = []
  for img in data:
    keep = img['id'] in all_split_ids and len(img['regions']) > 0
    if keep:
      new_data.append(img)
  return new_data

def lines_from(f):
    if not os.path.exists(f):
        return []
    lines = []
    for line in open(f).readlines():
        lines.append(line.strip('\n'))
    return lines

def main(args):

  # Get file path
  annopath = args.datadir + 'Annotations/%s.xml'
  imgpath = args.datadir + 'JPEGImages/%s.jpg'
  imgsetpath = args.datadir + 'ImageSets/Main/%s.txt'
  # read splits
  split_data = {}
  split_data['train'] = lines_from(args.train_split or imgsetpath %('train'))
  split_data['val'] = lines_from(args.val_split or imgsetpath %('val'))
  split_data['test'] = lines_from(args.test_split or imgsetpath %('test'))

  # get all annotations
  data = getAllAnnotations(annopath, imgpath, split_data['train'] + split_data['val'])

  all_data = [img['id'] for img in data] + split_data['test']

  # create the output hdf5 file handle
  f = h5py.File(args.h5_output, 'w')

  # add split information
  split = encode_splits(all_data, split_data)
  f.create_dataset('split', data=split)

  # build class label mapping
  cls_to_idx, idx_to_cls = build_cls_dict() # both mappings are dicts
  
  # encode labels
  labels_matrix, difficult = encode_labels(data, cls_to_idx)
  f.create_dataset('labels', data=labels_matrix)
  f.create_dataset('difficult', data=difficult)
  
  # encode boxes
  boxes_matrix = encode_boxes(data)
  f.create_dataset('boxes', data=boxes_matrix)
  
  # integer mapping between image ids and box ids
  img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
  f.create_dataset('img_to_first_box', data=img_to_first_box)
  f.create_dataset('img_to_last_box', data=img_to_last_box)
  filename_to_idx, idx_to_filename = build_filename_dict(all_data, imgpath)
  box_to_img = encode_filenames(data, imgpath, filename_to_idx)
  f.create_dataset('box_to_img', data=box_to_img)
  f.close()

  # and write the additional json file 
  json_struct = {
    'cls_to_idx': cls_to_idx,
    'idx_to_cls': idx_to_cls,
    'filename_to_idx': filename_to_idx,
    'idx_to_filename': idx_to_filename,
  }
  with open(args.json_output, 'w') as f:
    json.dump(json_struct, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # INPUT settings
  parser.add_argument('--datadir',
      default='/home/ruotian/code/fast-rcnn-torch/data/datasets/voc_2012/VOC2012/',
      help='The directory of data')
  # OUTPUT settings
  parser.add_argument('--json_output',
      default='data/voc12.json',
      help='Path to output JSON file')
  parser.add_argument('--h5_output',
      default='data/voc12.h5',
      help='Path to output HDF5 file')

  # OPTIONS
  parser.add_argument('--train_split',
      default=None, type=str,
      help='Split file for train') 
  parser.add_argument('--val_split',
      default=None, type=str,
      help='Split file for val') 
  parser.add_argument('--test_split',
      default=None, type=str,
      help='Split file for test') 
  args = parser.parse_args()
  main(args)

