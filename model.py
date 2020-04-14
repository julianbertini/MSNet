import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib

path_to_labels = "../../Task01_BrainTumour/labelsTr"
path_to_imgs = "../../Task01_BrainTumour/imagesTr"

AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_nifti(img_path_bytes, label=False):
    """ Reads in an nii.gz format image 

        Params:
            img_path_bytes - bytes, the image path in byte form
        Returns:
            img - the image in a tf.float32 nD array (image) or tf.uint8 nD array (label)
    """

    # Decode img_path from bytes to string
    img_path = img_path_bytes.decode("utf-8")
    
    # Load .nii.gz image and convert to tf.float32 dtype
    img = nib.load(img_path)
    img = np.asarray(img.dataobj)

    if label:
        img = tf.image.convert_image_dtype(img, tf.uint8)
    else:
        img = tf.image.convert_image_dtype(img, tf.float32)

    return img

def get_label_path(img_path):
    """ Gets the label image corresponding to the given image path
        
        Params:
            img_path - tf.Tensor, representing the path to an image file
        Returns:
            label_path - tf.Tensor, representing the path to the label file 
    """
    parts = tf.strings.split(img_path, os.path.sep)
    label_path = tf.strings.join([path_to_labels, parts[-1]], os.path.sep)

    return label_path


def process_image_train(img_path: tf.Tensor):
    """ * Callback function for tf.data.Dataset.map to process each image file path.
        * For each image file path, it returns a corresponding (image, label) pair.
        * This is the parent function that wraps all other processing helpers.
        * This is the processing for the training data specifically.

        Params:
            img_path - tf.Tensor, representing the path to an image file
        Returns:
            img, label - tuple of (tf.float32, tf.uint8) arrays representing the image and label arrays
    """

    label_path = get_label_path(img_path)

    img = read_nifti(img_path.numpy())
    label = read_nifti(label_path.numpy(), label=True)

    return img, label
    

def normalize(input_image):
    """ Normalizes the image data
        
        TODO: maybe split this into separate methods for training, testing, etc.
    """
    # Cast image to float32 type and normalize to [0,1] pixel values
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image, input_mask

#@tf.function
#def load_image_train(data)

def main():
    
    img_dir = pathlib.Path(path_to_imgs)
    
    img_count = len(list(data_dir.glob("*.nii.gz")))
 
    # The tf.data.Dataset API supports writing descriptive and efficient input pipelines.
    #   - Create a source dataset from your input data
    #   - Apply dataset transformation to preprocess the data
    #   - Iterate over the dataset and process the elements
    # Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory at once. 
    dataset = tf.data.Dataset

    # Generates a dataset list of all files matching one or more glob patters.
    # It will return filenames in a non-deterministic random shuffle order.
    img_ds = dataset.list_files(str(img_dir/"*"))
    
    
    # `num_parallel_calls` allows for mltiple images to be loaded/processed in parallel. 
    # the tf.py_function allows me to convert each element in `img_ds` to numpy format, which is necessary to read nifti images.
    # we have to do this b/c tensorflow does not have a custom .nii.gz image decoder like jpeg or png.
    # According to docs, this lowers performance, but I think this is still better than just doing a for loop b/c of the asynchronous
    train = img_ds.map(lambda x: tf.py_function(func=process_path, inp=[x], Tout=(tf.float32, tf.uint8)) , num_parallel_calls=AUTOTUNE)
    
    for image, label in train.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy().shape)
        print(np.nanmin(label.numpy()))



if __name__ == "__main__":
    main()
