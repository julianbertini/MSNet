import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib
from viz import Visualize

class WNet(tf.keras.Model):

    def __init__(self):
        """ Builds the WNet model for whole tumor segmentation
        """
        super(WNet, self).__init__(name='')

        self.block_1_a = ResidualIdentityBlock()

        self.block_1_b = ResidualIdentityBlock()



class ResidualIdentityBlock(tf.keras.Model):

    def __init__(self, kernels, n_output_chns, dilation_rates, with_residual=True):
        """ 
            Params:
                kernel - [..., [depth, height, width], ...], 2D array, where the number of rows represents 
                the number of "sub" blocks in the identity block, and each row is 3 integers denoting the dimensions of the conv. kernel
               
                n_output_chns - tuple, with 2 integer values denoting the number of output channels for each conv.

                dilation_rates - [..., [dilation in depth dim, dilation in height dim, dilation in width dim], ...], 2D array,
                mirroring the kernel array, where it indicates the dilations in depth, height, width, for each kernel.
        """

        super(ResidualIdentityBlock, self).__init__(name='')
        
        self.kernels = kernels
        self.dilation_rates = dilation_rates

        self.strides = [1,1,1]
        self.reg_decay = 1e-7
        self.n_output_chns = n_output_chns
        self.with_residual = with_residual 
        
        # 1) 3D convolution layer
        self.conv3a = tf.keras.layers.Conv3D(
                    self.n_output_chns, 
                    self.kernels[0], 
                    self.strides, 
                    padding="SAME", 
                    dilation_rate=self.dilation_rates[0],
                    use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    bias_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    name="conv3a")

        # 2) Batch normalization layer
        self.bn3a = tf.keras.layers.BatchNormalization(
                    beta_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    gamma_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    name="batch3a")
            
        # 3) PReLU activation function
        self.acti3a = tf.keras.layers.PReLU(name="acti3a")
        
        # 4) 3D convolution layer
        self.conv3b = tf.keras.layers.Conv3D(
                    self.n_output_chns, 
                    self.kernels[1], 
                    self.strides, 
                    padding="SAME", 
                    dilation_rate=self.dilation_rates[1],
                    use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    bias_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    name="conv3b")

        # 5) Batch normalization layer
        self.bn3b = tf.keras.layers.BatchNormalization(
                    beta_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    gamma_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    name="batch3b")
        
        # 6) PReLU activation function
        self.acti3b = tf.keras.layers.PReLU(name="acti3b")
        


    def call(self, input_tensor, training=False):
        
        # Connect layers

        # 1st sub-block
        output_tensor = self.conv3a(input_tensor)
        output_tensor = self.bn3a(output_tensor, training=training)
        output_tensor = self.acti3a(output_tensor)
        
        # 2nd sub-block
        output_tensor = self.conv3b(output_tensor)
        output_tensor = self.bn3b(output_tensor, training=training)

        # Add residual inputs  
        if self.with_residual: 
            output_tensor += input_tensor

        # Apply final activation
        output_tensor = self.acti3b(output_tensor)

        return output_tensor


def main():

    block = ResidualIdentityBlock([[1,3,3], [1,3,3]], 2, [[1,1,1], [1,1,1]])
    
    _ = block(tf.zeros([1, 10, 10, 10, 1]))

    print(block.layers)
    print(block.summary())



if __name__ == "__main__":
    main()
