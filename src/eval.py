import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import mkdir
import pathlib
from shutil import rmtree
from PIL.Image import open, fromarray

from viz import Visualize
from data_preprocessor import *
from model import *

def save(root_path, image, filename):
	print('saving image with shape:', image.shape)
	print('to:', join(root_path, filename))
	np.save(join(root_path, filename), image)
  

def dice_score(seg, ref):
	seg = tf.cast(seg, dtype=tf.float32)
	ref = tf.cast(ref, dtype=tf.float32)

	dsc_numerator = tf.constant(2, dtype=tf.float32) * tf.reduce_sum(tf.multiply(seg, ref)) + 1e-10
	dsc_denominator = tf.reduce_sum(seg) + tf.reduce_sum(ref) + 1e-10

	return tf.math.divide(dsc_numerator, dsc_denominator)

def dice_similarity(model, dataset=None, num=99):
	# Create test directory to save images
	save_dir = join('/home/jbertini/scratch-midway2/Python/MSNet/src','test_patches')
	if not exists(save_dir):
		mkdir(save_dir)
	else:
		rmtree(save_dir)
		mkdir(save_dir)

	infer = model.signatures['serving_default']
	if dataset:
		idx = 0
		scores = []
		for image, label in dataset.take(num):
			try:
				# Make prediction on entire batch
				pred_labels = infer(image)
				pred_labels = pred_labels[list(pred_labels.keys())[0]]
				# For each image in batch
				for b in range(label.shape[0]):
					image_b = image[b,:,:,:,0] # flair channel
					label_b = label[b]
					pred_b = pred_labels[b]
					# Determine prediction and calculate dice score
					pred_b = tf.nn.softmax(pred_b)
					pred_b = tf.argmax(pred_b, axis=-1)
					pred_b = tf.expand_dims(pred_b, axis=-1)
					scores.append(dice_score(pred_b, label_b).numpy())
					
					label_b = label_b[:,:,:,0]	
					pred_b = pred_b[:,:,:,0]
					# Save predictions and image patches to folder
					save(save_dir, pred_b.numpy(), 'pred_' + str(idx) + '_' + str(b))
					save(save_dir, label_b.numpy(), 'label_' + str(idx) + '_' + str(b))
					save(save_dir, image_b.numpy(), 'image_' + str(idx) + '_' + str(b))
				idx += 1
			except Exception as e:
				print("passing...check for errors")
				print(e)
				pass
		avg_score = sum(scores)/len(scores)
		print("Average dice score across " + str(len(scores)) + " test images: " + str(avg_score))
	else:
		print('Please provide a dataset.')

if __name__ == "__main__":

	dp = DataPreprocessor(tumor_region="whole tumor")

	val_img_dir = pathlib.Path(dp.path_to_val_imgs)

	dataset = tf.data.Dataset

	val_img_ds = dataset.list_files(str(val_img_dir/"*"))
	val_dataset = dp.prepare_for_testing(val_img_ds, purpose="val")

	#model = tf.keras.models.load_model('tf_model_v1.0')
	#model = tf.keras.models.load_model('tf_model_v1')
	saved_dir = "/home/jbertini/scratch-midway2/Python/MSNet/src/training_log/run_1/tf_MSNet_v2"
	model = tf.saved_model.load(saved_dir)
	dice_similarity(model, val_dataset, num=2)
