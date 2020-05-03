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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  run_eagerly=True)

    #tf.keras.utils.plot_model(model, show_shapes=True)


    model_history = model.fit(train_dataset, 
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              callbacks=[DisplayCallback()])

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
