import numpy as np
import glob
import argparse
from scipy import ndimage
import matplotlib.pyplot as plt
from nibabel import load
from os.path import join, split, exists
from os import mkdir
from shutil import rmtree
from sys import exit
from natsort import natsorted

from viz import Visualize

"""
This program pre-processes 3-D volumetric images to extract and save 
ROIs of the images automatically.

The goal is to focus in on the region of interest (e.g. brain) and remove
some of the background to improve network training.

This should be run BEFORE anythin else, as this will process the .nii.gz
images and save new ones for the ROIs. Those new images can then be used
for training and fed into the rest of the data preprocessing pipeline that
takes action when the training process begins.

TODO:
- Vectorize the process instead of looping through images to improve speed
"""

def check_edges(im, roi, slice_x, slice_y, padding):
  
  while sum(roi[0,:]) > 0 or \
        sum(roi[roi.shape[0]-1,:]) > 0 or \
        sum(roi[:,0]) > 0 or \
        sum(roi[:,roi.shape[1]-1]) > 0:  
    
    # Check x-dimension
    if sum(roi[0,:]) > 0:
      slice_x = slice(slice_x.start - 1, slice_x.stop)
      roi = im[slice_x, slice_y]
    if sum(roi[roi.shape[0]-1,:]) > 0:
      slice_x = slice(slice_x.start, slice_x.stop+1)
      roi = im[slice_x, slice_y]
    
    # Check y-dimension
    if sum(roi[:,0]) > 0: 
      slice_y = slice(slice_y.start-1, slice_y.stop)
      roi = im[slice_x, slice_y]
    if sum(roi[:,roi.shape[1]-1]) > 0:  
      slice_y = slice(slice_y.start, slice_y.stop+1)
      roi = im[slice_x, slice_y]
  
  # Add specified padding  
  slice_x = slice(slice_x.start-padding[0], slice_x.stop+padding[0])
  slice_y = slice(slice_y.start-padding[1], slice_y.stop+padding[1])
  roi = im[slice_x, slice_y]

  return roi, slice_x, slice_y

def _crop_roi_xy(im, padding):
  mask = im > im.mean()
  label_im, nb_labels = ndimage.label(mask)
  # Find the largest connected component
  sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
  mask_size = sizes < 10000
  remove_pixel = mask_size[label_im]
  label_im[remove_pixel] = 0
  labels = np.unique(label_im)
  label_im = np.searchsorted(labels, label_im)
  
  # Now that we have only one connected component, extract it's bounding box
  try:
    slice_x, slice_y = ndimage.find_objects(label_im==1)[0]
    slice_x = slice(slice_x.start, slice_x.stop)
    slice_y = slice(slice_y.start, slice_y.stop)
    roi = im[slice_x, slice_y]
    
    roi, slice_x, slice_y = check_edges(im, roi, slice_x, slice_y, padding)
    
    return roi, slice_x, slice_y
  except IndexError:
    # No objects were found; prob. minimal objects in image. Move on.
    return np.array([]), None, None 

def crop_roi(input_im, padding):
  """ Crop ROI around brain region
  """

  max_dims = [[1e10, -1],[1e10, -1]]
  max_z = [-1, -1]
  # reduce along the channel dimension should be registered
  im = np.sum(input_im, -1) 
  # Iterate along the depth dimension
  # change index to 0 for tensorflow depth dim order 
  depth_i = 2
  x_i = 0
  y_i = 1
  #print(im.shape)
  for i in range(im.shape[depth_i]):
    im_xy = im[:,:,i]

    if np.sum(im_xy) > 0:

      if max_z[0] == -1:
        max_z[0] = i

      roi_xy, slice_x, slice_y= _crop_roi_xy(im_xy, (padding[0],padding[1]))
      if roi_xy.any() and slice_x and slice_y:
        if slice_x.start < max_dims[0][0]:
          max_dims[0][0] = slice_x.start
        if slice_x.stop > max_dims[0][1]:
          max_dims[0][1] = slice_x.stop

        if slice_y.start < max_dims[1][0]:
          max_dims[1][0] = slice_y.start
        if slice_y.stop > max_dims[1][1]:
          max_dims[1][1] = slice_y.stop

    elif max_z[0] != max_z[1] and i > max_z[1]: 
      # have to add 1 cuz upper bound is exclusive
      max_z[1] = i + 1
  
  
  # Check that RIO x-y dims are within total image dims 
  max_dims[0][0] = max_dims[0][0] if max_dims[0][0] > 0 else 0
  max_dims[0][1] = max_dims[0][1] if max_dims[0][1] <= im.shape[x_i] else im.shape[x_i]
  max_dims[1][0] = max_dims[1][0] if max_dims[1][0] > 0 else 0
  max_dims[1][1] = max_dims[1][1] if max_dims[1][1] <= im.shape[y_i] else im.shape[y_i]
  slice_x = slice(max_dims[0][0], max_dims[0][1])
  slice_y = slice(max_dims[1][0], max_dims[1][1])
  
  # Check that ROI z-dims are within total image dims 
  padded_z_0 = max_z[0]-padding[2]
  padded_z_1 = max_z[1]+padding[2]
  max_z[0] =  padded_z_0 if padded_z_0 > 0 else 0  
  max_z[1] = padded_z_1 if padded_z_1 <= im.shape[depth_i] else im.shape[depth_i]
  
  # Re-create 3-D volume from ROIs 
  chn_stacks = []
  for j in range(input_im.shape[-1]):
    z_stacks = []
    for i in range(max_z[0], max_z[1]):
      z_stacks.append(input_im[:,:,i,j][slice_x, slice_y])
    chn_stacks.append(np.stack(z_stacks, axis=-1))
  roi_xyzc = np.stack(chn_stacks, axis=-1)
    
  #print("ROI dims:", max_dims, max_z)  
  
  return roi_xyzc, slice_x, slice_y, max_z

def crop_label_roi(label, slice_x, slice_y, max_z):
  z_stacks = []
  for i in range(max_z[0], max_z[1]):
    z_stacks.append(label[:,:,i][slice_x, slice_y])
  label_roi = np.stack(z_stacks, axis=-1)
  
  return label_roi



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--imageDir', help='relative/full path to image directory')
  parser.add_argument('--labelDir', help='relative/full path to label directory')
  parser.add_argument('--destDir', help='relative/full path to destination root directory')
  args = parser.parse_args()
  
  if args.imageDir:
    image_dir = args.imageDir
  if args.labelDir:
    label_dir = args.labelDir
  if args.destDir:
    dest_dir = args.destDir
    image_dest = join(dest_dir, 'p_' + str(split(image_dir)[-1]))
    label_dest = join(dest_dir, 'p_' + str(split(label_dir)[-1]))
    print('Saving processed images inside: ', dest_dir)
    print('Image ROI destination: ', image_dest)
    print('Label ROI destination: ', label_dest)
    
    if not exists(image_dest):
      mkdir(image_dest)
    else:
      rmtree(image_dest)
      mkdir(image_dest)
    if not exists(label_dest):
      mkdir(label_dest)
    else:
      rmtree(label_dest)
      mkdir(label_dest)
  else:
    print('Please provide a destination directory.')
    exit(0)


  images = glob.glob(image_dir + '/*.*')
  images = natsorted(images)
  labels = glob.glob(label_dir + '/*.*')
  labels = natsorted(labels)
  
  for i in range(len(images)):
    im = np.asarray(load(images[i]).dataobj)
    lb = np.asarray(load(labels[i]).dataobj)
    image_roi, slice_x, slice_y, max_z = crop_roi(im, (2,2,2))
    label_roi = crop_label_roi(lb, slice_x, slice_y, max_z)
    print('Image ROI dims: ', image_roi.shape)    
    print('Label ROI dims: ', label_roi.shape)    
    image_name = split(images[i])[-1].split('.')[0] 
    label_name = split(labels[i])[-1].split('.')[0] 
    np.save(join(image_dest,'p' + image_name), image_roi)
    np.save(join(label_dest,'p' + label_name), label_roi)
    print('Saved: ', join(image_dest,'p' + image_name))
    print('Saved: ', join(label_dest,'p' + label_name))

  #viz = Visualize()
  #viz.multi_slice_viewer([roi[:,:,:,0], label[:,:,:]])
  #plt.show()



