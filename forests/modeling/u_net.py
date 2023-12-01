# Basics imports
import numpy as np

# Tensor flow imports
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Rescaling, Conv2DTranspose
from tensorflow.keras import Model, Input


def encoder_mini_block_v1(inputs, 
                     n_filters = 32, 
                     dropout_prob = 0.2, 
                     max_pooling = True):
    """Create a convolution encoding mini block to chain in order to create U net models
       The v1 works with a padding = 'same' in the conv2D layers"""
    
    # First conv layer of the mini block
    conv = Conv2D(n_filters, 
                  kernel_size = (3, 3),  # filter size
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'HeNormal')(inputs)
    
    # Second conv layer of the mini block
    conv = Conv2D(n_filters, 
                  kernel_size = (3, 3),  # filter size
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'HeNormal')(conv)
    
    # Batch normalization layer to help deep models learn better
    conv = BatchNormalization()(conv, training = False)
    
    # Add dropout layer if dropout is input
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    
    # Add max pooling layer if true
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2, 2))(conv)
    
    
    else:
        next_layer = conv
    
    # Create the layers to return
    skip_connection = conv
    
    return next_layer, skip_connection


def decoder_mini_block(prev_layer_input, 
                     skip_layer_input, 
                     n_filters = 32):
    """Create a convolution decoding mini block to chain in order to create U net models"""
    
    # Create the up-conv layer
    up = Conv2DTranspose(n_filters, 
                         kernel_size = (3, 3), 
                         strides = (2, 2), 
                         padding = 'same')(prev_layer_input)
    
    # Merge the up-conv layer output with the skip layer input from the corresponding encoding mini-block
    merge = tf.concat([up, skip_layer_input], axis = 3)
    
    # First conv layer of the mini block
    conv = Conv2D(n_filters, 
                  kernel_size = (3, 3), 
                  activation = 'relu', 
                  padding = 'same', 
                  kernel_initializer = 'HeNormal')(merge)
    
    # Second conv layer of the mini block
    conv = Conv2D(n_filters, 
                  kernel_size = (3, 3), 
                  activation = 'relu', 
                  padding = 'same', 
                  kernel_initializer = 'HeNormal')(conv)
    
    return conv


