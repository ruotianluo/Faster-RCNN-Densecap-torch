# coding=utf8

import argparse, os, json, string
from collections import Counter
from Queue import Queue
from threading import Thread, Lock

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize

"""
This file expects a JSON file containing ground-truth regions and captions
in the same format as the region descriptions file from the Visual Genome
website. Concretely, this is a single large JSON file containing a list;
each element of the list describes a single image and has the following
format:

image{
"id" : int,
"width" : int,
"height" : int,
"file_name" : str,
"license" : int,
"flickr_url" : str,
"coco_url" : str,
"date_captured" : datetime,
}

annotation{
"id" : int,
"image_id" : int,
"category_id" : int,
"segmentation" : RLE or [polygon],
"area" : float,
"bbox" : [x,y,width,height],
"iscrowd" : 0 or 1,
}

We assume that all images are on disk in a single folder, and that
the filename for each image is the same as its id with a .jpg extension.

This file will be preprocessed into an HDF5 file and a JSON file with
some auxiliary information. The captions will be tokenized with some
basic preprocessing (split by words, remove special characters).

Note, in general any indices anywhere in input/output of this file are 1-indexed.

The output JSON file is an object with the following elements:
- cls_to_idx: Dictionary mapping strings to integers for encoding tokens, 
                in 1-indexed format.
- filename_to_idx: Dictionary mapping string filenames to indices.
- idx_to_cls: Inverse of the above.
- idx_to_filename: Inverse of the above.
- idx_to_cocoid
- cocoid_to_idx

The output HDF5 file has the following format to describe N images with
M total regions:

- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (x, y, w, h) where y and x are left up coordinates of the box,
  and are one-indexed.
- iscrowd: int32 array of shape (M,) giving whether the region is crowded or not
- labels: int32 array of shape (M,) giving the class label index  for each region.
  To recover a class label from an integer in this matrix,
  use idx_to_cls from the JSON output file.
- img_to_first_box: int32 array of shape (N,). If img_to_first_box[i] = j then
  captions[j] and boxes[j] give the first annotation for image i
  (using one-indexing).
- img_to_last_box: int32 array of shape (N,). If img_to_last_box[i] = j then
  captions[j] and boxes[j] give the last annotation for image i
  (using one-indexing).
- box_to_img: int32 array of shape (M,). If box_to_img[i] = j then then
  regions[i] and captions[i] refer to images[j] (using one-indexing).
"""

def build_class_dict(data):
  cls_to_idx, idx_to_cls = {}, {}
  cidx_to_idx = {}

  idx_to_cls[1] = '__background__'
  cls_to_idx['__background__'] = 1
  next_idx = 2

  for cat in data['categories']:
    cls_to_idx[cat['name']] = next_idx
    idx_to_cls[next_idx] = cat['name']
    cidx_to_idx[cat['id']] = next_idx
    next_idx = next_idx + 1

  for img in data['images']:
    for region in img['regions']:
      region['category_id'] = cidx_to_idx[region['category_id']]
  
  return cls_to_idx, idx_to_cls

def encode_labels(data, cls_to_idx):
  encoded_list = []
  iscrowd = []
  for img in data:
    for region in img['regions']:
      encoded_list.append(region['category_id'])
      iscrowd.append(region['iscrowd'])
  return np.asarray(encoded_list, dtype=np.int32), np.asarray(iscrowd, dtype=np.int32)

def encode_boxes(data):
  all_boxes = []
  for i, img in enumerate(data):
    for region in img['regions']:
      if region['category_id'] is None: continue
      # recall: x,y are 0-indexed
      x, y = region['bbox'][0] + 1, region['bbox'][1] + 1
      w, h = region['bbox'][2], region['bbox'][3]

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
    for region in img['regions']:
      if region['category_id'] is None: continue
      box_idx += 1
    img_to_last_box[img_idx - 1] = box_idx - 1 # -1 to make these inclusive limits
    img_idx += 1
  
  return img_to_first_box, img_to_last_box

def build_filename_dict(data):
  # First make sure all filenames
  filenames_list = [img['file_name'] for img in data]
  assert len(filenames_list) == len(set(filenames_list))
  
  next_idx = 1
  filename_to_idx, idx_to_filename = {}, {}
  for img in data:
    filename = img['file_name']
    filename_to_idx[filename] = next_idx
    idx_to_filename[next_idx] = filename
    next_idx += 1
  return filename_to_idx, idx_to_filename

def build_cocoid_dict(data):
  # First make sure all filenames
  cocoid_list = [img['id'] for img in data]
  assert len(cocoid_list) == len(set(cocoid_list))
  
  next_idx = 1
  cocoid_to_idx, idx_to_cocoid= {}, {}
  for img in data:
    cocoid = img['id']
    cocoid_to_idx[cocoid] = next_idx
    idx_to_cocoid[next_idx] = cocoid
    next_idx += 1
  return cocoid_to_idx, idx_to_cocoid

def encode_filenames(data, filename_to_idx):
  filename_idxs = []
  for img in data:
    filename = img['file_name']
    idx = filename_to_idx[filename]
    for region in img['regions']:
      if region['category_id'] is None: continue
      filename_idxs.append(idx)
  return np.asarray(filename_idxs, dtype=np.int32)

def get_filepath(s, image_dir):
  if 'train' in s:
    return os.path.join(image_dir, s[s.find('train'):s.find('train') + 9], s)
  if 'val' in s:
    return os.path.join(image_dir, s[s.find('val'):s.find('val') + 7], s)

def encode_splits(data, split_data):
  """ Encode splits as intetgers and return the array. """
  lookup = {'train': 0, 'val': 1, 'test': 2}
  id_to_split = {}
  split_array = np.zeros(len(data['images']))
  for split, idxs in split_data.iteritems():
    for idx in idxs:
      id_to_split[idx] = split
  for i, img in enumerate(data['images']):
    if id_to_split[img['id']] in lookup:
      split_array[i] = lookup[id_to_split[img['id']]]
  return split_array


def filter_images(data, split_data):
  """ Keep only images that are in some split and have some captions """
  all_split_ids = set()
  for split_name, ids in split_data.iteritems():
    all_split_ids.update(ids)
  tmp_data = []
  for img in data['images']:
    keep = img['id'] in all_split_ids and len(img['regions']) > 0
    if keep:
      tmp_data.append(img)
  new_data = {}
  new_data['images'] = tmp_data
  new_data['categories'] = data['categories']
  return new_data

def make_data(args):
  data = {}
  train_data = json.load(open(args.region_data %('train')))
  val_data = json.load(open(args.region_data %('val')))

  data['images'] = train_data['images'] + val_data['images']
  data['annotations'] = train_data['annotations'] + val_data['annotations']

  # Merge all the regions in the key 'images'.
  tmp_data = {}
  for anno in data['annotations']:
    tmp_data[anno['image_id']] = tmp_data.get(anno['image_id'], []) + [anno]

  for img in data['images']:
    img['regions'] = tmp_data.get(img['id'], [])
    img['file_name'] = get_filepath(img['file_name'], args.image_dir)

  del data['annotations']
  data['categories'] = train_data['categories']

  return data

def main(args):

  # read in the data
  data = make_data(args)
  with open(args.split_json, 'r') as f:
    split_data = json.load(f)

  # Only keep images that are in a split
  print 'There are %d images total' % len(data['images'])
  data = filter_images(data, split_data)
  print 'After filtering for splits there are %d images' % len(data['images'])

  # create the output hdf5 file handle
  f = h5py.File(args.h5_output, 'w')

  # add split information
  split = encode_splits(data, split_data)
  f.create_dataset('split', data=split)

  # build class label mapping
  cls_to_idx, idx_to_cls = build_class_dict(data) # both mappings are dicts

  # Remove the redundant category information
  data = data['images']

  # encode labels
  labels_matrix, iscrowd_vector = encode_labels(data, cls_to_idx)
  f.create_dataset('labels', data=labels_matrix)
  f.create_dataset('iscrowd', data=iscrowd_vector)
  
  # encode boxes
  boxes_matrix = encode_boxes(data)
  f.create_dataset('boxes', data=boxes_matrix)
  
  # integer mapping between image ids and box ids
  img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
  f.create_dataset('img_to_first_box', data=img_to_first_box)
  f.create_dataset('img_to_last_box', data=img_to_last_box)
  filename_to_idx, idx_to_filename = build_filename_dict(data)
  box_to_img = encode_filenames(data, filename_to_idx)
  f.create_dataset('box_to_img', data=box_to_img)
  f.close()

  cocoid_to_idx, idx_to_cocoid = build_cocoid_dict(data)

  # and write the additional json file 
  json_struct = {
    'cocoid_to_idx': cocoid_to_idx,
    'idx_to_cocoid': idx_to_cocoid,
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
  parser.add_argument('--region_data',
      default='/home/ruotian/code/pycoco/annotations/instances_%s2014.json',
      help='Input JSON file with regions and captions')
  parser.add_argument('--image_dir',
      default='/home/ruotian/data/MSCOCO/',
      help='Directory containing all images')
  parser.add_argument('--split_json',
      default='info/coco_splits.json',
      help='JSON file of splits')

  # OUTPUT settings
  parser.add_argument('--json_output',
      default='data/COCO.json',
      help='Path to output JSON file')
  parser.add_argument('--h5_output',
      default='data/COCO.h5',
      help='Path to output HDF5 file')

  args = parser.parse_args()
  main(args)

