import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib
from viz import Visualize

#from train import *

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 4
BUFFER_SIZE = 385
# Right now, the smaller label patch will be centered along the same center point as
# the image patch. So the label patch will be missing what's left over from the image patch
# on either side equally.
IMG_PATCH_SIZE = [19, 124, 124, 4]  # [depth, height, width, channels]
LABEL_PATCH_SIZE = [11, 124, 124, 1]  # [depth, height, width, channels]

# These are correct order. Obtained from Coursera.
MODALITIES = {"flair": 0, "t1": 1, "t1c": 2, "t2": 3}
TUMOR_REGIONS = {"whole tumor": 1, "tumor core": 2, "active tumor": 3}

# TODO: create brain weight_map and pass it along attached to label. This will be used in training to omit background from training loss.

class DataPreprocessor():

    def __init__(self, 
                 tumor_region=None,
                 path_to_train_labels="../../../Task01_BrainTumour/p_labelsTr",
                 path_to_train_imgs="../../../Task01_BrainTumour/p_imagesTr",
                 path_to_val_imgs="../../../Task01_BrainTumour/p_imagesVal",
                 path_to_val_labels="../../../Task01_BrainTumour/p_labelsVal",
                 path_to_test_imgs="../../../Task01_BrainTumour/imagesTest",
                 path_to_test_labels="../../../Task01_BrainTumour/labelsTest"):

        self.path_to_train_imgs = path_to_train_imgs
        self.path_to_train_labels = path_to_train_labels

        self.path_to_test_imgs = path_to_test_imgs
        self.path_to_test_labels = path_to_test_labels

        self.path_to_val_imgs = path_to_val_imgs
        self.path_to_val_labels = path_to_val_labels

        self.tumor_region = tumor_region
    
    def get_weight_map(self, image_patch, label_patch):
      """
      Creates a binary mask around the brain region to be used by loss
      function such as to only consider brain region in computing loss.

      image and label passed in should be PRE normalization, so that 
      the background is still zero.
      """
      weight_map = tf.where(image_patch > tf.constant(0, dtype=tf.float32), tf.ones_like(image_patch, dtype=tf.dtypes.int32), tf.zeros_like(image_patch, dtype=tf.dtypes.int32))
      
      return weight_map

    def read_npy(self, img_path_bytes, label=False, img_channels=4):
        """ Reads from a numpy saved file
        """
        # Decode img_path from bytes to string
        img_path = img_path_bytes.decode("utf-8")

        img = np.load(img_path)

        if label:
            img = tf.image.convert_image_dtype(img, tf.int32)
            img = tf.reshape(
                img, [img.shape[0], img.shape[1], img.shape[2], 1])
            img = tf.transpose(img, perm=(2, 1, 0, 3))
        else:
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.transpose(img, perm=(2, 1, 0, 3))
            img = img[:, :, :, 0:img_channels]

        return img

    def read_nifti(self, img_path_bytes, label=False, img_channels=4):
        """ Reads in an nii.gz format image

            NOTE: Image is read in as [x, y, z, chn], but later we treat it as [z, y, x, chn] due to
                  tf convention, so here the image is reshaped to match [z, y, x, chn]

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
            img = tf.image.convert_image_dtype(img, tf.int32)
            img = tf.reshape(
                img, [img.shape[0], img.shape[1], img.shape[2], 1])
            img = tf.transpose(img, perm=(2, 1, 0, 3))
        else:
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.transpose(img, perm=(2, 1, 0, 3))
            img = img[:, :, :, 0:img_channels]

        return img

    def get_label_path(self, img_path, purpose="train"):
        """ Gets the label image corresponding to the given image path

            Params:
                img_path - tf.Tensor, representing the path to an image file
            Returns:
                label_path - tf.Tensor, representing the path to the label file
        """
        if purpose == "train": 
            label_path = self.path_to_train_labels
        elif purpose == "val":
            label_path = self.path_to_val_labels
        elif purpose == "test":
            label_path = self.path_to_test_labels

        parts = tf.strings.split(img_path, os.path.sep)
        label_path = tf.strings.join([label_path, parts[-1]], os.path.sep)

        return label_path

    def map_path_to_patch_pair(self, img_path, purpose):
        """ * Maps an image path to a patch pair (image and corresponding label patches)
            * Wrapper for the `process_image_train` function 
            * Sets the shape of output from tf.py_function, which addressess a known issue with passing Dataset objects to 
              keras model.fit function

            Params:
                img_patch - denoting the file path to the image 
            Returns:
                image_patch, label_patch - the image, label patch pair as Tensors
        """
        image_patch, label_patch = tf.py_function(func=self.process_image_train, inp=[
                                                  img_path, purpose], Tout=(tf.float32, tf.int32))

        image_shape = IMG_PATCH_SIZE
        label_shape = LABEL_PATCH_SIZE
        image_patch.set_shape(image_shape)
        label_patch.set_shape(label_shape)

        return image_patch, label_patch


    # TODO:
    # Right now, I am only returning a single (image_patch, label_patch) pair. Let's define this as one "training unit" (TU).
    # But to the best of my understanding, the way things work with patches is that each TU should not be only a sinlge patch pair,
    # but rather it should be a collection of patch pairs from the original (image, label) pair. This makes sense since we want to try
    # to fully represent the image in a collection of patches (meanwhile maintaining the benefits of training on smaller sub-units).

    # So here I should modify this to actually return a collection of patch pairs (I don't know the perfect number yet, maybe test this).
    # This also agrees with what the Matlab model does, which is return 16 patch pairs per image. And each unit inside a batch, the TU, is
    # actually a collection of patches. So a minibatch size of 5 would actually contain 16*5 = 80 patch pairs.
    #
    # Update: based on MSNet Github, it seemes like they only use 1 random patch per image for training.

    def process_image_train(self, img_path: tf.Tensor, purpose):
        """ * Callback function for tf.data.Dataset.map to process each image file path.
            * For each image file path, it returns a corresponding (image, label) pair.
            * This is the parent function that wraps all other processing helpers.
            * This is the processing for the training data specifically.

            Params:
                img_path - tf.Tensor, representing the path to an image file
                purpose - "train" for training; "val" for validation; "test" for testing
            Returns:
                img, label - tuple of (tf.float32, tf.uint8) arrays representing the image and label arrays
        """

        label_path = self.get_label_path(img_path, purpose)

        input_image = self.read_npy(img_path.numpy(), img_channels=4)
        input_label = self.read_npy(label_path.numpy(), label=True)
        
        #input_image = self.read_nifti(img_path.numpy(), img_channels=4)
        #input_label = self.read_nifti(label_path.numpy(), label=True)

        # Make label binary for tumor region in question
        if self.tumor_region:
            input_label = tf.where(input_label >= tf.constant(TUMOR_REGIONS[self.tumor_region], dtype=tf.int32), tf.constant(1, dtype=tf.int32), tf.constant(0, dtype=tf.int32))

        # Fetch random image patch
        image_patch, label_patch = self.get_random_patch(
            input_image, input_label)
        
        weight_map = self.get_weight_map(image_patch, label_patch)

        # Normalize image patch AFTER creating the patch
        image_patch = self.normalize(image_patch)

        if purpose == "train":
            # Augment Data
            image_patch, label_patch = self.augment_patch(
                image_patch, label_patch)


        return image_patch, label_patch

    def augment_patch(self, image_patch, label_patch):
        """ 
           TODO: add up/down flipping with tf.image.flip_up_down 
        """
        # Give image a random brightness
        #image_patch = tf.image.random_brightness(image_patch, max_delta=0.5)

        # Randomly perform either rotation or reflection

        rand_val = tf.random.uniform(())
        if rand_val <= 0.25:
            image_patch = self.apply_transform(image_patch, tf.image.rot90)
            label_patch = self.apply_transform(label_patch, tf.image.rot90)
        elif rand_val <= 0.5:
            image_patch = self.apply_transform(
                image_patch, tf.image.flip_left_right)
            label_patch = self.apply_transform(
                label_patch, tf.image.flip_left_right)
        
        # Random brightness and random contrast transformations 
        #image_patch = self.apply_transform(image_patch, lambda input_img: tf.image.random_contrast(input_img, 0.3, 0.5))
        #image_patch = self.apply_transform(image_patch, lambda input_img: tf.image.random_brightness(input_img, 0.2)) 

        return image_patch, label_patch

    def apply_transform(self, input_img, tf_transform):
        """ applies a 2D tf transform (i.e., rotate) to a 3D image by iterating through
            each slice

            Params:
                image - the tf array detailing the image
                tf_transform - the tf transform function, i.e., tf.image.rot90
            Returns:
                image_trans - transformed 3D image
        """

        img_size = tf.shape(input_img).numpy()

        z_slices = []
        for idx in range(img_size[0]):
            trans_sl = tf_transform(input_img[idx, :,:,:])
            z_slices.append(trans_sl)

        output_img = tf.stack(z_slices, axis=0)

        return output_img

    def get_random_patch(self, input_image, input_label):
        """ * Wrapper for the __get_image_patch helper to fetch a random image and label patch
            * Loops until a patch is returned where the labels are not all zeros (all background)

            Params:
                input_image - tf.float32 array representing the entire image 
                input_label - tf.float32 array representing the entire labeled image
            Returns:
                image_patch - tf.float32 array representing the image patch 
                label_path -  tf.float32 array representing the corresponding label patch
        """
        max_tries = 1000 # prevent infinite loop
        tries = 0
        flag = False
        while (flag == False and tries < max_tries):
            image_patch, prev_center = self.__get_image_patch(
                input_image, IMG_PATCH_SIZE[:-1])
            label_patch, _ = self.__get_image_patch(
                input_label, LABEL_PATCH_SIZE[:-1], is_label=True, prev_center=prev_center)
            tumor_pixels = tf.where(label_patch > 0, tf.ones_like(label_patch, dtype=tf.dtypes.int32), tf.zeros_like(label_patch, dtype=tf.dtypes.int32))
            tumor_pixels = tf.math.reduce_sum(tumor_pixels)
            tumor_pixels = tf.cast(tumor_pixels, dtype=tf.float32)
            total_pixels = tf.size(label_patch, out_type=tf.dtypes.int32)
            total_pixels = tf.cast(total_pixels, dtype=tf.float32)
            tumor_frac = tf.math.divide(tumor_pixels, total_pixels)
            
            # Keep trying until tumor region is at least 5% of the total patch
            assert total_pixels > tumor_pixels
            if tumor_frac > 0.05:
                #print(tumor_frac)
                flag = True

            tries += 1

        return image_patch, label_patch

    def __get_image_patch(self, input_image, patch_size, is_label=False, prev_center=None):
        """ * Extracts a patch from the given image/label pair of dims [patch_size, patch_size, 4]
            * The last dimension is the number of modalities (4 in this case)

            Params:
                input_image - tf.float32 array representing an image 
                input_label - tf.float32 array representing the labeled image
                center - [depth, height, width], array specifying center to use instead of random
            Returns:
                image_patch - tf.float32 array representing the image patch 
                label_path -  tf.float32 array representing the corresponding label patch
        """
        remainder_z = 0
        remainder_y = 0
        remainder_x = 0

        half_margin_z = int(patch_size[0] / 2)
        half_margin_y = int(patch_size[1] / 2)
        half_margin_x = int(patch_size[2] / 2)

        # The center point on each axis
        center_z = int(np.floor(tf.shape(input_image)[0] / 2))
        center_y = int(np.floor(tf.shape(input_image)[1] / 2))
        center_x = int(np.floor(tf.shape(input_image)[2] / 2))

        if patch_size[0] % 2 != 0:
            remainder_z = 1
        if patch_size[1] % 2 != 0:
            remainder_y = 1
        if patch_size[2] % 2 != 0:
            remainder_x = 1

        # Generate a random center point on each axis for the patch. The random center point is limited to only
        # values within a sub-cube inside the image cube where a PATCH_SIZE patch is able to be created
        if prev_center is None:
            # Depth dimension
            rand_z = np.random.randint(
                center_z - (center_z - half_margin_z), center_z + (center_z - half_margin_z - remainder_z) + 1)
            # Height dimension
            rand_y = np.random.randint(
                center_y - (center_y - half_margin_y), center_y + (center_y - half_margin_y - remainder_y) + 1)
            # Width dimension
            rand_x = np.random.randint(
                center_x - (center_x - half_margin_x), center_x + (center_x - half_margin_x - remainder_x) + 1)
        else:
            rand_z = prev_center[0]
            rand_y = prev_center[1]
            rand_x = prev_center[2]

        # Make sure dimensions and margins work out correctly
        assert half_margin_x + remainder_x <= center_x and half_margin_y + \
            remainder_y <= center_y and half_margin_z + remainder_z <= center_z

        # Index into the input_image to extract the patch
        if is_label:
            image_patch = input_image[rand_z-half_margin_z:rand_z+half_margin_z+remainder_z,
                                rand_y-half_margin_y:rand_y+half_margin_y+remainder_y,
                                      rand_x-half_margin_x:rand_x+half_margin_x+remainder_x]
        else:
            image_patch = input_image[rand_z-half_margin_z:rand_z+half_margin_z+remainder_z,
                                rand_y-half_margin_y:rand_y+half_margin_y+remainder_y,
                                rand_x-half_margin_x:rand_x+half_margin_x+remainder_x, :]

        return image_patch, [rand_z, rand_y, rand_x]

    def prepare_for_testing(self, img_ds, purpose):
        """ * Takes in the testing/validation dataset and prepares images for testing/validation
            * Calls `process_image_train` to process each image, label pair 
            * This is the wrapper function to do all data proceessing to ready it for testing. 

            Params:
                img_ds - tf.data.Dataset, containing the paths to the training images 
                cache - Bool or String, denoting whether to cache in memory or in a directory
            Returns:
                dataset - tf.data.Dataset, a shuffled batch of image/label pairs of size BATCH_SIZE ready for training 
        """
        # `num_parallel_calls` allows for mltiple images to be loaded/processed in parallel.
        # the tf.py_function allows me to convert each element in `img_ds` to numpy format, which is necessary to read nifti images.
        # we have to do this b/c tensorflow does not have a custom .nii.gz image decoder like jpeg or png.
        # According to docs, this lowers performance, but I think this is still better than just doing a for loop b/c of the asynchronous
        dataset = img_ds.map(lambda img_path: self.map_path_to_patch_pair(
            img_path, purpose=purpose))

        # Select a batch of size BATCH_SIZE
        dataset = dataset.batch(BATCH_SIZE)

        return dataset

    def prepare_for_training(self, img_ds, cache=True, shuffle_buffer_size=BUFFER_SIZE):
        """ * Takes in the entire dataset and prepares shuffled batches of images for training
            * Calls `process_image_train` to process each image, label pair 
            * This is the wrapper function to do all data proceessing to ready it for training. 

            Params:
                img_ds - tf.data.Dataset, containing the paths to the training images 
                cache - Bool or String, denoting whether to cache in memory or in a directory
                shuffle_buffer_size - denoting the size of each collection of shuffled images (default 1000)
            Returns:
                dataset - tf.data.Dataset, a shuffled batch of image/label pairs of size BATCH_SIZE ready for training 
        """
        # `num_parallel_calls` allows for mltiple images to be loaded/processed in parallel.
        # the tf.py_function allows me to convert each element in `img_ds` to numpy format, which is necessary to read nifti images.
        # we have to do this b/c tensorflow does not have a custom .nii.gz image decoder like jpeg or png.
        # According to docs, this lowers performance, but I think this is still better than just doing a for loop b/c of the asynchronous
        dataset = img_ds.map(lambda img_path: self.map_path_to_patch_pair(
            img_path, purpose="train"), num_parallel_calls=None)
        # dataset = img_ds.map(lambda x: tf.py_function(func=self.process_image_train, inp=[x], Tout=(tf.float32, tf.uint8)),
        #        num_parallel_calls=AUTOTUNE)

        # So caching works by either:
        #   a) caching given dataset object in memory (small dataset)
        #   b) caching given dataset object in a specified directory (large dataset -- doesn't fit in memory)
        if cache:
            # If entire dataset does not fit in memory, then specify cache as the name of directory to cache data into
            if isinstance(cache, str):
                print("Caching on disk...")
                dataset = dataset.cache(cache)
            # If entire dataset fits in memory, then just cache dataset in memory
            else:
                dataset = dataset.cache()


        # Select a batch of size BATCH_SIZE
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(BATCH_SIZE)

        # Repeat this dataset indefinitely; meaning, we never run out of data to pull from.
        # Since we called shuffle() before, each repetition of the data here will be a differently
        # shuffled collection of images (shuffle() by default reshuffles after each iteration, and
        # repeat() is basically calling an indefinite number of iterations)
        dataset = dataset.repeat()

        # prefetch allows later elements to be prepared while the current element is being processed.
        # Improves throughput at the expense of using additional memory to store prefetched elements.
        #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def normalize(self, input_image):
        """ * Normalizes the image data by subtracting the mean and dividing by the stdev per modality 
            * All images should be normalized, including test images

            Params:
                input_image - tf.float32 array representing the image, dims: (depth, height, width, channels)

            TODO: * Maybe split this into separate methods for training, testing, etc.
                  * Also, Matlab model sets all pixels that were 0 before normalization to 0 after normalization.
                  * I'm not sure entirely why this is done or what the effects are. Not going to do it for now.
                  * Normalize pixel values to be between [0,1] within each modality 
                  * Matlab code also forces all pixels to be between [-5, 5] (before normalizing to be between [0,1])
                    and this just seems kind of arbitrary. I'm not sure what this does. 
        """
        
        # Normalize across each z-slice and each modality (as they do in Coursera)
        c_stacks = [] 
        for c in range(input_image.shape[3]):
          z_stacks = []
          for z in range(input_image.shape[0]):
          
            image_slice = input_image[z,:,:,c]
            centered = image_slice - tf.math.reduce_mean(image_slice)  
            
            std = tf.math.reduce_std(centered)
            if std != 0:
              centered_scaled = centered / std
              z_stacks.append(centered_scaled)
            else:
              z_stacks.append(centered)
          
          c_stacks.append(tf.stack(z_stacks, axis=0))
        norm_image = tf.stack(c_stacks, axis=-1)
      
        #print(tf.math.reduce_std(norm_image[50,:,:,2]))
  
        #input_image = tf.transpose(input_image, perm=(2, 1, 0, 3))  

        ## Calculate the mean for the image for each modality
        ## mean is an array with 4 elements: the mean for each modality (or "sequence")
        #mean = tf.math.reduce_mean(input_image, [0, 1])
        ## Same for standard deviation
        #std = tf.math.reduce_std(input_image, [0, 1])

        ## Subtract the mean from each element and divide by standard deviation
        #input_image = tf.math.subtract(input_image, mean)
        #input_image = tf.where(std != 0, tf.divide(input_image,std), input_image)

        ## Set image values to range from [0,1]

        ### Add min to make all elements >= 0
        ##min_per_mod = tf.math.reduce_min(input_image, [0, 1,2])
        ### subtract since min is negative, and we want to add
        ##input_image = tf.math.subtract(input_image, min_per_mod)
        ### Divide by max to make all elements <= 1
        ##max_per_mod = tf.math.reduce_max(input_image, [0, 1,2])
        ##input_image = tf.math.divide(input_image, max_per_mod)

        #input_image = tf.transpose(input_image, perm=(2, 1, 0, 3))

        return norm_image


def main():
    """ Use for testing/debugging purposes
    """

    dp = DataPreprocessor(tumor_region="whole tumor")

    img_dir = pathlib.Path(dp.path_to_train_imgs)

    # The tf.data.Dataset API supports writing descriptive and efficient input pipelines.
    #   - Create a source dataset from your input data
    #   - Apply dataset transformation to preprocess the data
    #   - Iterate over the dataset and process the elements
    # Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory at once.
    dataset = tf.data.Dataset

    # Generates a dataset list of all files matching one or more glob patters.
    # It will return filenames in a non-deterministic random shuffle order.
    img_ds = dataset.list_files(str(img_dir/"*"))

    # Create batches, shuffle, etc
    train = dp.prepare_for_training(img_ds)

    # Visualizing 3D volumes
    viz = Visualize()
  
    for i in range(10):
      for image, label in train.take(1):
          for z in range(image.shape[1]):
            img_slice = image[3,z,:,:,2]
            std = tf.math.reduce_std(image)
            mean = tf.math.reduce_mean(image)
            #print(std)
            #print(mean)

          viz.multi_slice_viewer([image.numpy()[0, :,:,:, 0], label.numpy()[0,:,:,:, 0]])
          plt.show()


if __name__ == "__main__":
    main()
