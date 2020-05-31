import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib

from viz import Visualize
from data_preprocessor import *
from model import *

# I'm actually guessing at this order... I think it's right 
MODALITIES = {"t1": 0, "t1c": 1, "t2": 2, "flair": 3}

class DisplayCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

class GeneralizedDiceLoss(tf.keras.losses.Loss):

    def call(self, ground_truth, prediction, weight_map=None):
        """
            `weight_map` represents same thing as `loss_weight` in tf
            except that we apply it here directly instead of passing in 
            through the model.fit attribute `loss_weight`
        """
        prediction = tf.cast(prediction, tf.float32)
        if len(ground_truth.shape) == len(prediction.shape):
            ground_truth = ground_truth[..., -1]
        one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

        if weight_map is not None:
            num_classes = prediction.shape[1].value
            weight_map_nclasses = tf.tile(
                tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
            ref_vol = tf.sparse_reduce_sum(
                weight_map_nclasses * one_hot, reduction_axes=[0])

            intersect = tf.sparse_reduce_sum(
                weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
            seg_vol = tf.reduce_sum(
                tf.multiply(weight_map_nclasses, prediction), 0)
        else:
            ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
            intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                             reduction_axes=[0])
            seg_vol = tf.reduce_sum(prediction, 0)
        if type_weight == 'Square':
            weights = tf.reciprocal(tf.square(ref_vol))
        elif type_weight == 'Simple':
            weights = tf.reciprocal(ref_vol)
        elif type_weight == 'Uniform':
            weights = tf.ones_like(ref_vol)
        else:
            raise ValueError("The variable type_weight \"{}\""
                             "is not defined.".format(type_weight))
        new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                           tf.reduce_max(new_weights), weights)
        generalised_dice_numerator = \
            2 * tf.reduce_sum(tf.multiply(weights, intersect))
        generalised_dice_denominator = tf.reduce_sum(
            tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
        generalised_dice_score = \
            generalised_dice_numerator / generalised_dice_denominator
        generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                          generalised_dice_score)
        return 1 - generalised_dice_score


def main():

    dp = DataPreprocessor(tumor_region="whole tumor")
    
    img_dir = pathlib.Path(dp.path_to_imgs)
    
    TRAIN_LENGTH = len(list(img_dir.glob("*.nii.gz")))
    print(TRAIN_LENGTH) 
    # The tf.data.Dataset API supports writing descriptive and efficient input pipelines.
    #   - Create a source dataset from your input data
    #   - Apply dataset transformation to preprocess the data
    #   - Iterate over the dataset and process the elements
    # Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory at once. 
    dataset = tf.data.Dataset

    # Generates a dataset list of all files matching one or more glob patters.
    # It will return filenames in a non-deterministic random shuffle order.
    img_ds = dataset.list_files(str(img_dir/"*"))
    
    # Takes in the imgs paths and does all data preprocesesing, returning shuffled batches ready for training 
    train_dataset = dp.prepare_for_training(img_ds)
    
    # Setup training
    EPOCHS = 5
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    model = MSNet("test_msnet")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=GeneralizedDiceLoss(),
                  metrics=['accuracy'],
                  run_eagerly=True)

    #tf.keras.utils.plot_model(model, show_shapes=True)


    model_history = model.fit(train_dataset, 
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH)

    # Visualize a patch
    #viz = Visualize()

    #for image, label in train_dataset.take(1):
    #    print("Image shape: ", image.shape)
    #    print("Label: ", label.shape)
    #    print(np.nanmax(label.numpy()))    
    #    
    #    viz.multi_slice_viewer([image.numpy()[0,:,:,:,0], label.numpy()[0,:,:,:]])
    #    plt.show()


if __name__ == "__main__":
    main()
