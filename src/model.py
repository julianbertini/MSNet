import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import nibabel as nib
from viz import Visualize

from data_preprocessor import BATCH_SIZE
OUTPUT_CHNS = 32
LARGER_NETS = ['WNET', 'TNET']

class MSNet(tf.keras.Model):

    def __init__(self, name, num_classes=2):
        """ Builds the WNet model for whole tumor segmentation

            TODO: add support for ENet. Right now, only supports WNet and TNet

            Params:
                name - string, denoting either WNET, TNET, or ENET.
        """
        super(MSNet, self).__init__(name=name.upper())

        self.num_classes = num_classes
        self.reg_decay = 1e-7

        # Residual Block Layers

        self.block_1_a = ResidualIdentityBlock("res_block_1_a")
        self.block_1_b = ResidualIdentityBlock("res_block_1_b")

        self.block_2_a = ResidualIdentityBlock("res_block_2_a")
        self.block_2_b = ResidualIdentityBlock("res_block_2_b")

        self.block_3_a = ResidualIdentityBlock("res_block_3_a")
        self.block_3_b = ResidualIdentityBlock(
            "res_block_3_b", dilation_rates=[[1, 2, 2], [1, 2, 2]])
        self.block_3_c = ResidualIdentityBlock(
            "res_block_3_c", dilation_rates=[[1, 3, 3], [1, 3, 3]])

        self.block_4_a = ResidualIdentityBlock(
            "res_block_4_a", dilation_rates=[[1, 3, 3], [1, 3, 3]])
        self.block_4_b = ResidualIdentityBlock(
            "res_block_4_b", dilation_rates=[[1, 2, 2], [1, 2, 2]])
        self.block_4_c = ResidualIdentityBlock("res_block_4_c")

        # Fusion Layers

        self.fusion_1 = FusionBlock("fusion_1")
        self.fusion_2 = FusionBlock("fusion_2")
        self.fusion_3 = FusionBlock("fusion_3")
        self.fusion_4 = FusionBlock("fusion_4")

        # Downsampling Layers

        self.down_block_1 = DownsampleBlock("down_1")
        self.down_block_2 = DownsampleBlock("down_2")

        # Upsampling Layers
        # need to make an extra prediction block for the ENet, which does not have upsampling for the first prediction

        self.up_block_1 = UpsampleBlock("up_1")

        self.up_block_2_a = UpsampleBlock(
            "up_2_a", n_output_chns=self.num_classes*2)
        self.up_block_2_b = UpsampleBlock(
            "up_2_b", n_output_chns=self.num_classes*2)

        self.up_block_3_a = UpsampleBlock(
            "up_block_3_a", n_output_chns=self.num_classes*4)
        self.up_block_3_b = UpsampleBlock(
            "up_block_3_b", n_output_chns=self.num_classes*4)

        # Central Slice Layers

        self.central_slice_1 = CentralSliceBlock("central_1", 2)
        self.central_slice_2 = CentralSliceBlock("central_2", 1)
		
		# ENET-specific layer
        self.pred1 = tf.keras.layers.Conv3D(
					self.num_classes,
                    kernel_size=[1, 3, 3],
                    padding = 'SAME',
                    kernel_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    bias_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                    name='pred1')

        # Final Prediction Layer

        self.final_pred = tf.keras.layers.Conv3D(
            self.num_classes,
            kernel_size=[1, 3, 3],
            padding="SAME",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg_decay),
            bias_regularizer=tf.keras.regularizers.l2(self.reg_decay),
            name="final_pred")

    def call(self, input_tensor, training=False):

        output_tensor_1 = self.block_1_a(input_tensor, training=training)
        output_tensor_1 = self.block_1_b(output_tensor_1, training=training)
        output_tensor_1 = self.fusion_1(output_tensor_1, training=training)

        if self.name in LARGER_NETS: 
            output_tensor_1 = self.down_block_1(output_tensor_1, training=training)

        output_tensor_1 = self.block_2_a(output_tensor_1, training=training)
        output_tensor_1 = self.block_2_b(output_tensor_1, training=training)
        output_tensor_1 = self.fusion_2(output_tensor_1, training=training)

        output_tensor_2 = self.down_block_2(output_tensor_1, training=training)
        output_tensor_2 = self.block_3_a(output_tensor_2, training=training)
        output_tensor_2 = self.block_3_b(output_tensor_2, training=training)
        output_tensor_2 = self.block_3_c(output_tensor_2, training=training)
        output_tensor_2 = self.fusion_3(output_tensor_2, training=training)

        output_tensor_3 = self.block_4_a(output_tensor_2, training=training)
        output_tensor_3 = self.block_4_b(output_tensor_3, training=training)
        output_tensor_3 = self.block_4_c(output_tensor_3, training=training)
        output_tensor_3 = self.fusion_4(output_tensor_3, training=training)

        ### Predictions Path ###

        # OUTPUT 1
        output_tensor_1 = self.central_slice_1(output_tensor_1)
        if self.name in LARGER_NETS: 
            output_tensor_1 = self.up_block_1(output_tensor_1, training=training)
        else: #here add other option for ENET
            output_tensor_1 = self.pred1(output_tensor_1)

        # OUTPUT 2
        output_tensor_2 = self.central_slice_2(output_tensor_2)
        output_tensor_2 = self.up_block_2_a(output_tensor_2, training=training)
        if self.name in LARGER_NETS: 
            output_tensor_2 = self.up_block_2_b(output_tensor_2, training=training)

        # OUTPUT 3
        output_tensor_3 = self.up_block_3_a(output_tensor_3, training=training)
        if self.name in LARGER_NETS: 
            output_tensor_3 = self.up_block_3_b(output_tensor_3, training=training)

        ### Combine 3 Outputs ###
        concat = tf.concat([output_tensor_1, output_tensor_2,
                            output_tensor_3], axis=-1, name="final_concat")
        pred = self.final_pred(concat)

        return pred


class CentralSliceBlock(tf.keras.Model):
    """ Grabs the central part of a tensor along the depth dimension
    """

    def __init__(self, name, margin):

        super(CentralSliceBlock, self).__init__(name=name)

        self.margin = margin

    def call(self, input_tensor):
        input_shape = input_tensor.get_shape().as_list()

        if input_shape[0] is None:
            input_shape = [BATCH_SIZE] + input_shape[1:]
            input_tensor.set_shape(input_shape)

        begin = [0]*len(input_shape)
        begin[1] = self.margin
        #begin = tf.convert_to_tensor(begin, dtype=tf.int32)

        output_shape = input_shape
        output_shape[1] = output_shape[1] - 2 * self.margin
        #output_shape = tf.convert_to_tensor(output_shape, dtype=tf.int32)

        # print(input_shape)
        # print(input_tensor.shape)
        output_tensor = tf.slice(
            input_tensor, begin, output_shape, name="slice")

        return output_tensor


class UpsampleBlock(tf.keras.Model):

    def __init__(self, name, kernel=[1, 3, 3], strides=[1, 2, 2], n_output_chns=OUTPUT_CHNS):

        super(UpsampleBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        self.kernel = kernel
        self.strides = strides
        self.reg_decay = 1e-7
        self.dilation_rate = [1, 1, 1]

        # 1) 3D transpose convolution layer
        self.conv3a = tf.keras.layers.Conv3DTranspose(
            self.n_output_chns,
            self.kernel,
            self.strides,
            padding="SAME",
            dilation_rate=self.dilation_rate,
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

    def call(self, input_tensor, training=False):

        # Connect layers
        output_tensor = self.conv3a(input_tensor)
        output_tensor = self.bn3a(output_tensor, training=training)
        output_tensor = self.acti3a(output_tensor)

        return output_tensor


class DownsampleBlock(tf.keras.Model):

    def __init__(self, name, kernel=[1, 3, 3], strides=[1, 2, 2], n_output_chns=OUTPUT_CHNS):

        super(DownsampleBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        self.reg_decay = 1e-7

        # 1) 3D convolution layer
        self.conv3a = tf.keras.layers.Conv3D(
            self.n_output_chns,
            kernel,
            strides,
            padding="SAME",
                    dilation_rate=[1, 1, 1],
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

    def call(self, input_tensor, training=False):

        # Connect layers
        output_tensor = self.conv3a(input_tensor)
        output_tensor = self.bn3a(output_tensor, training=training)
        output_tensor = self.acti3a(output_tensor)

        return output_tensor


class FusionBlock(tf.keras.Model):

    def __init__(self, name, kernel=[3, 1, 1], strides=[1, 1, 1], n_output_chns=OUTPUT_CHNS):
        """ * The fusion block, which is just one convolutional layer, batch norm, and activation.
            * I think of this as fusing the outputs from the previous residual blocks  
        """
        super(FusionBlock, self).__init__(name=name)

        self.reg_decay = 1e-7

        # 1) 3D convolution layer
        self.conv3a = tf.keras.layers.Conv3D(
            filters=n_output_chns,
            kernel_size=kernel,
            strides=strides,
            padding="valid",
                    dilation_rate=[1, 1, 1],
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

    def call(self, input_tensor, training=False):

        # Connect layers
        output_tensor = self.conv3a(input_tensor)
        output_tensor = self.bn3a(output_tensor, training=training)
        output_tensor = self.acti3a(output_tensor)

        return output_tensor


class ResidualIdentityBlock(tf.keras.Model):

    def __init__(self, name, dilation_rates=[[1, 1, 1], [1, 1, 1]], kernels=[[1, 3, 3], [1, 3, 3]], n_output_chns=OUTPUT_CHNS, with_residual=True):
        """ 
            Params:
                * kernel - [..., [depth, height, width], ...], 2D array, where the number of rows represents 
                  the number of "sub" blocks in the identity block, and each row is 3 integers denoting the dimensions of the conv. kernel

                * n_output_chns - integer, denoting the number of output channels for each conv.

                * dilation_rates - [..., [dilation in depth dim, dilation in height dim, dilation in width dim], ...], 2D array,
                  mirroring the kernel array, where it indicates the dilations in depth, height, width, for each kernel.
        """

        super(ResidualIdentityBlock, self).__init__(name=name)

        self.kernels = kernels
        self.dilation_rates = dilation_rates

        self.strides = [1, 1, 1]
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
            output_tensor = AddResidual(
                bypass_flow=input_tensor)(output_tensor)

        # Apply final activation
        output_tensor = self.acti3b(output_tensor)

        return output_tensor


class AddResidual(tf.keras.Model):
    """
    This class takes care of the elementwise sum in a residual connection
    It matches the channel dims from two branch flows,
    by either padding or projection if necessary.
    """

    def __init__(self, bypass_flow, name='residual'):

        self.layer_name = name
        self.bypass_flow = bypass_flow

        super(AddResidual, self).__init__(name=self.layer_name)

    def infer_spatial_rank(self, input_tensor):
        """
        e.g. given an input tensor [Batch, X, Y, Z, Feature] the spatial rank is 3
        """
        input_shape = input_tensor.shape
        input_shape.with_rank_at_least(3)

        return int(input_shape.ndims - 2)

    def call(self, param_flow, training=False):
        bypass_flow = self.bypass_flow
        n_param_flow = param_flow.shape[-1]
        n_bypass_flow = bypass_flow.shape[-1]
        spatial_rank = self.infer_spatial_rank(param_flow)

        output_tensor = param_flow

        if n_param_flow > n_bypass_flow:  # pad the channel dim
            pad_1 = np.int((n_param_flow - n_bypass_flow) // 2)
            pad_2 = np.int(n_param_flow - n_bypass_flow - pad_1)
            padding_dims = np.vstack(([[0, 0]],
                                      [[0, 0]] * spatial_rank,
                                      [[pad_1, pad_2]]))
            bypass_flow = tf.pad(tensor=bypass_flow,
                                 paddings=padding_dims.tolist(),
                                 mode='CONSTANT')
        elif n_param_flow < n_bypass_flow:  # make a projection

            projector = tf.keras.layers.Conv3D(
                n_param_flow,
                kernel_size=1,
                strides=1,
                padding="SAME",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                bias_regularizer=tf.keras.regularizers.l2(self.reg_decay),
                name="projector")
            bypass_flow = projector(bypass_flow)

        # element-wise sum of both paths
        output_tensor = param_flow + bypass_flow

        return output_tensor


def main():

    # Testing the individual components

    input_tensor = tf.zeros([1, 19, 96, 96, 4])

    #block = ResidualIdentityBlock("test_res", [[1,1,1], [1,1,1]])
    #_ = block(input_tensor)

    #fuseBlock = FusionBlock("fuse_test")
    #_ = fuseBlock(input_tensor)

    #downBlock = DownsampleBlock("down_test")
    #_ = downBlock(input_tensor)

    #upBlock = UpsampleBlock("up_test")
    #_ = upBlock(input_tensor)

    #centralSlice = CentralSliceBlock("center_test", 2)
    #_ = centralSlice(input_tensor)

    wnet = MSNet(name="enet")
    pred = wnet(input_tensor)

    print(pred.shape)


if __name__ == "__main__":
    main()
