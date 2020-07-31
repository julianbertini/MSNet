import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib

from viz import Visualize
from data_preprocessor import *
from model import *


def __tp_map(seg, ref):
	"""
		This function calculates the true positive map (i.e. how many 
		reference voxels are positive)

		:return: TP map
	"""
	seg = tf.cast(seg, dtype=bool)
	ref = tf.cast(ref, dtype=bool)
	return tf.cast(tf.math.logical_and(ref, seg), dtype=tf.float32)

def __tp(tp_map):
	return tf.math.reduce_sum(tp_map)

def dice_score(seg, ref):
	tp_map = __tp_map(seg, ref)
	tp = __tp(tp_map)
	seg = tf.cast(seg, dtype=tf.float32)
	ref = tf.cast(ref, dtype=tf.float32)	
	numerator = tf.cast(tf.math.multiply(tf.constant(2, dtype=tf.float32), tp), dtype=tf.float32)
	denominator = tf.math.reduce_sum(tf.math.add(ref, seg))

	return tf.math.divide(numerator, denominator)

def dice_similarity(model, dataset=None, num=60):
	scores = []
	infer = model.signatures['serving_default']
	if dataset:
		for image, mask in dataset.take(num):
			try:
				pred_labels = infer(image)
				pred_labels = pred_labels[list(pred_labels.keys())[0]]
				pred_labels = tf.nn.softmax(pred_labels)
				pred_labels = tf.argmax(pred_labels, axis=-1)
				pred_labels = tf.expand_dims(pred_labels, -1)
				scores.append(dice_score(pred_labels, mask).numpy())
			except:
				pass
		avg_score = sum(scores)/len(scores)
		print("Average dice score across " + str(len(scores)) + " test images: " + str(avg_score))
	else:
		display([sample_image, sample_mask,
			create_mask(model.predict(sample_image[tf.newaxis, ...]))])

def main():
	
	dp = DataPreprocessor(tumor_region="whole tumor")

	train_img_dir = pathlib.Path(dp.path_to_train_imgs)
	val_img_dir = pathlib.Path(dp.path_to_val_imgs)

	dataset = tf.data.Dataset

	val_img_ds = dataset.list_files(str(val_img_dir/"*"))
	val_dataset = dp.prepare_for_testing(val_img_ds, purpose="val")

	#model = tf.keras.models.load_model('tf_model_v1.0')
	#model = tf.keras.models.load_model('tf_model_v1')
	model = tf.saved_model.load("test")
	dice_similarity(model, val_dataset)
	




if __name__ == "__main__":
	main()
