import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib
from viz import Visualize

path_to_labels = "../../Task01_BrainTumour/labelsTr"
path_to_imgs = "../../Task01_BrainTumour/imagesTr"

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 5
BUFFER_SIZE = 10
PATCH_SIZE = 144    

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

    input_image = read_nifti(img_path.numpy())
    input_label = read_nifti(label_path.numpy(), label=True)
    
    # Normalize images 
    normalize(input_image)
    
    # Create a patch for an image and the corresponding label image
    image_patch, label_patch = get_random_patch(input_image, input_label)
    
    return image_patch, label_patch

def get_random_patch(input_image, input_label):
    """ * Wrapper for the __get_image_patch helper to fetch a random image and label patch
        * Loops until a patch is returned where the labels are not all zeros (all background)

        Params:
            input_image - tf.float32 array representing the entireimage 
            input_label - tf.float32 array representing the entire labeled image
        Returns:
            image_patch - tf.float32 array representing the image patch 
            label_path -  tf.float32 array representing the corresponding label patch
    """

    flag = False 
    while (flag == False):
        image_patch, label_patch = __get_image_patch(input_image, input_label)
        if tf.math.reduce_sum(label_patch, [0,1,2]) > 0:
            flag = True

    return image_patch, label_patch

def __get_image_patch(input_image, input_label):
    """ * Extracts a patch from the given image/label pair of dims [patch_size, patch_size, 4]
        * The last dimension is the number of modalities (4 in this case)

        Params:
            input_image - tf.float32 array representing an image 
            input_label - tf.float32 array representing the labeled image
        Returns:
            image_patch - tf.float32 array representing the image patch 
            label_path -  tf.float32 array representing the corresponding label patch
    """
    
    half_margin = int(PATCH_SIZE / 2)
    
    # The center point on each axis
    center_x = int(np.floor(tf.shape(input_image)[0] / 2))
    center_y = int(np.floor(tf.shape(input_image)[1] / 2))
    center_z = int(np.floor(tf.shape(input_image)[2] / 2))
    
    # Generate a random center point on each axis for the patch. The random center point is limited to only
    # values within a sub-cube inside the image cube where a PATCH_SIZE patch is able to be created
    rand_x = np.random.randint(center_x - (center_x - half_margin), center_x + (center_x - half_margin) + 1)
    rand_y = np.random.randint(center_y - (center_y - half_margin), center_y + (center_y - half_margin) + 1)
    rand_z = np.random.randint(center_z - (center_z - half_margin), center_z + (center_z - half_margin) + 1)
    
    # Index into the input_image to extract the patch
    image_patch = input_image[rand_x-half_margin:rand_x+half_margin, 
                        rand_y-half_margin:rand_y+half_margin, 
                        rand_z-half_margin:rand_z+half_margin,:]

    label_patch = input_label[rand_x-half_margin:rand_x+half_margin, 
                        rand_y-half_margin:rand_y+half_margin, 
                            rand_z-half_margin:rand_z+half_margin]

    
    return image_patch, label_patch


def prepare_for_training(dataset, cache=True, shuffle_buffer_size=BUFFER_SIZE):
    """ Takes in the entire dataset and prepares shuffled batches of images for training
        
        Params:
            dataset - tf.data.Dataset, tensor representing a tuple of the training images and respective label images
            cache - Bool or String, denoting whether to cache in memory or in a directory
            shuffle_buffer_size - denoting the size of each collection of shuffled images (default 1000)
        Returns:
            dataset - tf.data.Dataset, a shuffled batch of image/label pairs of size BATCH_SIZE used to training 
    """
    # So caching works by either:
    #   a) caching given dataset object in memory (small dataset)
    #   b) caching given dataset object in a specified directory (large dataset -- doesn't fit in memory)
    if cache:
        # If entire dataset does not fit in memory, then specify cache as the name of directory to cache data into 
        if isinstance(cache, str):
            dataset = dataset.cache(cache) 
        # If entire dataset fits in memory, then just cache dataset in memory
        else:
            dataset = dataset.cache()
    
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Repeat this dataset indefinitely; meaning, we never run out of data to pull from.
    # Since we called shuffle() before, each repetition of the data here will be a differently
    # shuffled collection of images (shuffle() by default reshuffles after each iteration, and
    # repeat() is basically calling an indefinite number of iterations)
    dataset = dataset.repeat()
    
    dataset = dataset.batch(BATCH_SIZE)

    # prefetch allows later elements to be prepared while the current element is being processed. 
    # Improves throughput at the expense of using additional memory to store prefetched elements. 
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def normalize(input_image):
    """ * Normalizes the image data by subtracting the mean and dividing by the stdev per modality 
        * All images should be normalized, including test images
        
        Params:
            input_image - tf.float32 array representing the image, dims: (240, 240, 155, 4)
        
        TODO: * Maybe split this into separate methods for training, testing, etc.
              * Also, Matlab model sets all pixels that were 0 before normalization to 0 after normalization.
              * I'm not sure entirely why this is done or what the effects are. Not going to do it for now.
              * Normalize pixel values to be between [0,1] within each modality 
              * Matlab code also forces all pixels to be between [-5, 5] (before normalizing to be between [0,1])
                and this just seems kind of arbitrary. I'm not sure what this does. 
    """
    
    # Calculate the mean across all images for each modality
    # mean is an array with 4 elements: the mean for each modality (or "sequence")
    mean = tf.math.reduce_mean(input_image, [0,1,2])
    # Same for standard deviation
    std = tf.math.reduce_std(input_image, [0,1,2])
    
    # Subtract the mean from each element and divide by standard deviation
    input_image = tf.math.subtract(input_image, mean)
    input_image = tf.math.divide(input_image, std)
    

    return input_image



def main():
    
    img_dir = pathlib.Path(path_to_imgs)
    
    img_count = len(list(img_dir.glob("*.nii.gz")))
 
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
    image_label_pairs = img_ds.map(lambda x: tf.py_function(func=process_image_train, inp=[x], Tout=(tf.float32, tf.uint8)) , num_parallel_calls=AUTOTUNE)
    
    # Create batches, shuffle, etc  
    train = prepare_for_training(image_label_pairs)
    
    viz = Visualize()

    for image, label in train.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy().shape)
        print(np.nanmax(label.numpy()))    
        
        viz.multi_slice_viewer([image.numpy()[:,:,:,0], label.numpy()])
        plt.show()

if __name__ == "__main__":
    main()
