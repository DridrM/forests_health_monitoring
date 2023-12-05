# Basics imports
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Tensor flow imports
import tensorflow as tf

# tf.config.run_functions_eagerly(True)

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# Forests package imports
from forests.modeling.u_net_pipeline import make_generator_pipeline


###################
# Model constants #
###################

# For now, we hardcode some constants in order to make the model run properly.
# Don't hesitate to override these constants when you make your call of the 'compile_and_fit_u_net' function

# Define the variable that control the model
IMAGE_SIZE = (128, 128, 3)
TARGET_SIZE = IMAGE_SIZE[:-1]
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 5
METRICS = ['accuracy', tf.keras.metrics.Recall()]

# Define paths
DATASET_PATH = "../raw_data/ForestNetDataset/"
RAW_IMAGES_PATH = DATASET_PATH + "examples"
TRAIN_PATHS = [DATASET_PATH + 'train/features', DATASET_PATH + 'train/targets']
VALID_PATHS = [DATASET_PATH + 'valid/features', DATASET_PATH + 'valid/targets']
TEST_PATHS = [DATASET_PATH + 'test/features', DATASET_PATH + 'test/targets']


#########################
# Custom loss functions #
#########################

# Custom rmse loss
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Custom dice loss
def dice_coef(y_true, y_pred, smooth = 100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


############################
# Compile and fit function #
############################

# Define a function to compile and fit the model
def compile_and_fit_u_net(u_net_model, 
                          epochs, 
                          loss_function, 
                          train_paths = TRAIN_PATHS, 
                          valid_paths = VALID_PATHS, 
                          metrics = METRICS, 
                          patience = PATIENCE, 
                          batch_size = BATCH_SIZE, 
                          target_size = TARGET_SIZE, 
                          learning_rate = LEARNING_RATE, 
                          ):
    """Compile and fit the u-net model :
    - u_net_model : Tensorflow u-net model
    - train_paths : List of strings with first the feature images path and after the targets mask path
    - valid_paths : List of strings with first the feature images path and after the targets mask path
    - epochs : Integer
    - batch_size : Integer
    - target_size : tuple
    - learning_rate : Float
    - loss_function : Function object
    - metrics : List of strings
    Return model history
    """
    
    # 1/ Compile the model
    # Define the optimizer and its learning rate
    optimizer = optimizers.Adam(learning_rate)
    
    # Compile the model
    u_net_model.compile(optimizer = optimizer, 
                        loss = loss_function, 
                        metrics = metrics)
    
    # 2/ Fit the model
    # Compute the number of steps per epochs
    path_for_listing = train_paths[-1] # Take a path for input in the os.listdir func
    subfolder = path_for_listing.split('/')[-1] # Pick the last subfolder of the path
    steps_per_epoch = len(os.listdir(path_for_listing + '/' + subfolder)) // batch_size # Reconstruct the right path to count the number of images in the train directory
    
    # Compute the number of val steps per epochs (same logic as before)
    path_for_listing_val = valid_paths[-1]
    subfolder_val = path_for_listing_val.split('/')[-1]
    val_steps_per_epoch = len(os.listdir(path_for_listing_val + '/' + subfolder_val)) // batch_size
    
    # Construct the train generator
    train_gen = make_generator_pipeline(*train_paths, batch_size, target_size)
    
    # Construct the valid generator
    valid_gen = make_generator_pipeline(*valid_paths, batch_size, target_size)
    
    # Define the early stopping
    es = EarlyStopping(patience = patience, restore_best_weights = True)
    
    # Fit the model !
    history = u_net_model.fit(train_gen, 
                              steps_per_epoch = steps_per_epoch, 
                              epochs = epochs, 
                              validation_data = valid_gen, 
                              validation_steps = val_steps_per_epoch, 
                              callbacks = [es])
    
    return history


##################################
# Evaluate and predict functions #
##################################

# Define a function to evaluate on the test set
def evaluate_u_net(u_net_hitory, 
                   test_paths = TEST_PATHS, 
                   batch_size = BATCH_SIZE, 
                   target_size = TARGET_SIZE) -> tuple:
    """Evaluate a fitted u-net model based on the test data
    - u_net_history : history object of a model
    - test_paths : List of strings with first the feature images path and after the targets mask path
    - batch_size : Integer
    - target_size : tuple
    """
    
    # Construct the test pipeline
    test_gen = make_generator_pipeline(*test_paths, batch_size, target_size)

    # Pick up the trained model
    u_net_fitted = u_net_hitory.model

    # Compute the number of val steps per epochs (same logic as before)
    path_for_listing_test = test_paths[-1]
    subfolder_test = path_for_listing_test.split('/')[-1]
    test_steps_per_epoch = len(os.listdir(path_for_listing_test + '/' + subfolder_test)) // batch_size

    # Evaluate on the test set
    test_history = u_net_fitted.evaluate(test_gen, 
                            batch_size = batch_size, 
                            steps = test_steps_per_epoch, 
                            verbose = 1)
    
    return test_gen, u_net_fitted, test_history


# Define a function to predict and plot
def predict_and_plot_u_net(test_gen, 
                           u_net_fitted, 
                           batch_size = BATCH_SIZE, 
                           figsize = (15, 10)) -> None:
    """Predict and plot given a trained u-net model
    - test_gen : Generator object
    - u_net_fitted : A fitted u-net model
    - batch_size : Integer
    - figsize : tuple
    """

    try:
        # Generate a random number within the batch size
        rd_index = np.random.randint(0, batch_size)

        # Pick an image and a mask to predict
        image_to_predict = test_gen.__next__()[0][rd_index]
        mask_to_predict = test_gen.__next__()[1][rd_index]

        # Predict
        mask_predicted = u_net_fitted.predict(test_gen.__next__()[0])[rd_index]

        # Plot prediction alongside image and mask to predict
        plt.figure(figsize = figsize)

        # Plot the image ot predict
        plt.subplot(1, 3, 1)
        plt.imshow(image_to_predict.numpy().astype(np.float32))

        # Plot the mask to predict
        plt.subplot(1, 3, 2)
        plt.imshow(mask_to_predict.numpy().astype(np.float32), cmap = 'gray')

        # Plot the predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(mask_predicted, cmap = 'gray')

        plt.show()
    
    except:
        print("Batch out of bounds. Nothing to show here !")

