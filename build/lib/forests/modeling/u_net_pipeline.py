# Basics imports
import numpy as np
# import os
# import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Tensor flow imports
import tensorflow as tf

# tf.config.run_functions_eagerly(True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Forests package imports
# from forests.preprocessing.mask import pullFile, classifyData_Pickle
# from forests.modeling.u_net import encoder_mini_block_v1, decoder_mini_block_v1
# from forests.modeling.modeling import plot_history


# Define a scaling function for images and target
def scale_image_and_target(image, target):
    """Scale the image and its target mask"""
    
    # Scale from 0 to 1 the image
    image = tf.cast(image, tf.float16) / 255.0
    
    # Min max scale the mask target
    target = (tf.cast(target, tf.float16) - np.min(target)) / (np.max(target) - np.min(target))
    
    return image, target


# Define the pipeline function
def make_generator_pipeline(features_path: str, 
                        targets_papth: str, 
                        batch_size: int, 
                        target_size: tuple) -> None:
    """Construct an image and target generator for u-net models that feed the .fit method"""
    
    # Set the image generator with the tf method flow from directory
    image_generator = ImageDataGenerator().flow_from_directory(features_path, 
                                                          class_mode = None, 
                                                          color_mode = 'rgb', 
                                                          target_size = target_size, 
                                                          batch_size = batch_size, 
                                                          seed = 42)
    
    # Set the target mask generator with the tf method flow from directory
    mask_generator = ImageDataGenerator().flow_from_directory(targets_papth, 
                                                          class_mode = None, 
                                                          color_mode = 'grayscale', 
                                                          target_size = target_size, 
                                                          batch_size = batch_size, 
                                                          seed = 42)
    
    # Construct the generator object
    pipeline_generator = zip(image_generator, mask_generator)
    
    for (image, mask) in pipeline_generator:
        # Scale the image and its target mask
        (image, mask) = scale_image_and_target(image, mask)
        
        yield (image, mask)
