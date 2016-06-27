# coding=utf8

import argparse, os, json, string
from collections import Counter
from Queue import Queue
from threading import Thread, Lock

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize

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

- images: uint8 array of shape (N, 3, image_size, image_size) of pixel data,
  in BDHW format. Images will be resized so their longest edge is image_size
  pixels long, aligned to the upper left corner, and padded with zeros.
  The actual size of each image is stored in the image_heights and image_widths
  fields.
- image_heights: int32 array of shape (N,) giving the height of each image.
- image_widths: int32 array of shape (N,) giving the width of each image.
- original_heights: int32 array of shape (N,) giving the original height of
  each image.
- original_widths: int32 array of shape (N,) giving the original width of
  each image.
- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (xc, yc, w, h) where yc and xc are center coordinates of the box,
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

def getAllAnnotations(annopath, all_data):
  data = []
  for img_id in all_data:
    tmp = getAnnotations(annopath, img_id)
    tmp['id'] = img_id
    if not type(tmp['object']) is list:
      tmp['object'] = [tmp['object']]
    data.append(tmp)
  return data

def encode_labels(data, cls_to_idx):
  encoded_list = []
  difficult = []
  for i, img in enumerate(data):
    for region in img['object']:
      encoded_list.append(cls_to_idx[region['name']])
      difficult.append(region['difficult'])
  return np.asarray(encoded_list, dtype=np.int32), np.asarray(difficult, dtype=np.int32)

def encode_boxes(data, original_heights, original_widths, image_size):
  all_boxes = []
  xwasbad = 0
  ywasbad = 0
  wwasbad = 0
  hwasbad = 0
  for i, img in enumerate(data):
    H, W = original_heights[i], original_widths[i]
    scale = float(image_size) / max(H, W)
    for region in img['object']:
      if region['name'] is None: continue
      # recall: x,y are 1-indexed
      x, y = round(scale*(int(region['bndbox']['xmin'])-1)+1), round(scale*(int(region['bndbox']['ymin'])-1)+1)
      w = round(scale*(int(region['bndbox']['xmax'])-int(region['bndbox']['xmin'])))
      h = round(scale*(int(region['bndbox']['ymax'])-int(region['bndbox']['ymin'])))
      
      # clamp to image
      if x < 1: x = 1
      if y < 1: y = 1
      if x > image_size - 1: 
        x = image_size - 1
        xwasbad += 1
      if y > image_size - 1: 
        y = image_size - 1
        ywasbad += 1
      if x + w > image_size: 
        w = image_size - x
        wwasbad += 1
      if y + h > image_size: 
        h = image_size - y
        hwasbad += 1

      box = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32) # also convert to center-coord oriented
      assert box[2]>=0 # width height should be positive numbers
      assert box[3]>=0
      all_boxes.append(box)

  print 'number of bad x,y,w,h: ', xwasbad, ywasbad, wwasbad, hwasbad
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

def build_filename_dict(data):
  # First make sure all filenames
  filenames_list = [img['filename'] for img in data]
  assert len(filenames_list) == len(set(filenames_list))
  
  next_idx = 1
  filename_to_idx, idx_to_filename = {}, {}
  for img in data:
    filename = img['filename']
    filename_to_idx[filename] = next_idx
    idx_to_filename[next_idx] = filename
    next_idx += 1
  return filename_to_idx, idx_to_filename

def encode_filenames(data, filename_to_idx):
  filename_idxs = []
  for img in data:
    filename = img['filename']
    idx = filename_to_idx[filename]
    for region in img['object']:
      if region['name'] is None: continue
      filename_idxs.append(idx)
  return np.asarray(filename_idxs, dtype=np.int32)

def add_images(imgpath, all_data, h5_file, args):
  num_images = len(all_data) 
  
  shape = (num_images, 3, args.image_size, args.image_size)
  image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
  original_heights = np.zeros(num_images, dtype=np.int32)
  original_widths = np.zeros(num_images, dtype=np.int32)
  image_heights = np.zeros(num_images, dtype=np.int32)
  image_widths = np.zeros(num_images, dtype=np.int32)
  
  lock = Lock()
  q = Queue()
  
  for i, img_id in enumerate(all_data):
    filename = os.path.join(imgpath % img_id)
    q.put((i, filename))
    
  def worker():
    while True:
      i, filename = q.get()
      img = imread(filename)
      # handle grayscale
      if img.ndim == 2:
        img = img[:, :, None][:, :, [0, 0, 0]]
      H0, W0 = img.shape[0], img.shape[1]
      img = imresize(img, float(args.image_size) / max(H0, W0))
      H, W = img.shape[0], img.shape[1]
      # swap rgb to bgr. Is this the best way?
      r = img[:,:,0].copy()
      img[:,:,0] = img[:,:,2]
      img[:,:,2] = r

      lock.acquire()
      if i % 1000 == 0:
        print 'Writing image %d / %d' % (i, len(all_data))
      original_heights[i] = H0
      original_widths[i] = W0
      image_heights[i] = H
      image_widths[i] = W
      image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
      lock.release()
      q.task_done()
  
  print('adding images to hdf5.... (this might take a while)')
  for i in xrange(args.num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()

  h5_file.create_dataset('image_heights', data=image_heights)
  h5_file.create_dataset('image_widths', data=image_widths)
  h5_file.create_dataset('original_heights', data=original_heights)
  h5_file.create_dataset('original_widths', data=original_widths)

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
  all_data = split_data['train'] + split_data['val'] + split_data['test']

  # create the output hdf5 file handle
  f = h5py.File(args.h5_output, 'w')

  # add several fields to the file: images, and the original/resized widths/heights
  add_images(imgpath, all_data, f, args)

  # add split information
  split = encode_splits(all_data, split_data)
  f.create_dataset('split', data=split)

  # build class label mapping
  cls_to_idx, idx_to_cls = build_cls_dict() # both mappings are dicts
  
  # get all annotations
  data = getAllAnnotations(annopath, split_data['train'] + split_data['val'])
  
  # encode labels
  labels_matrix, difficult = encode_labels(data, cls_to_idx)
  f.create_dataset('labels', data=labels_matrix)
  f.create_dataset('difficult', data=difficult)
  
  # encode boxes
  original_heights = np.asarray(f['original_heights'])
  original_widths = np.asarray(f['original_widths'])
  boxes_matrix = encode_boxes(data, original_heights, original_widths, args.image_size)
  f.create_dataset('boxes', data=boxes_matrix)
  
  # integer mapping between image ids and box ids
  img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
  f.create_dataset('img_to_first_box', data=img_to_first_box)
  f.create_dataset('img_to_last_box', data=img_to_last_box)
  filename_to_idx, idx_to_filename = build_filename_dict(data)
  box_to_img = encode_filenames(data, filename_to_idx)
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
      default='data/voc12-regions-dicts.json',
      help='Path to output JSON file')
  parser.add_argument('--h5_output',
      default='data/voc12-regions.h5',
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
  parser.add_argument('--image_size',
      default=720, type=int,
      help='Size of longest edge of preprocessed images')  
  parser.add_argument('--num_workers', default=5, type=int)
  args = parser.parse_args()
  main(args)

