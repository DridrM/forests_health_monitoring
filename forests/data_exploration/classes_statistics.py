# Basic imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tensorflow imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plot_classes_distrib(data_path: str, 
                         classes_names: list, 
                         image_shape = (350, 350, 3), 
                         ratio = 0.01, 
                         channel = 1) -> None:
    """Plot the distribution of the means of each"""
    
    
    # Setting batch size
    nb_in_classes = [] # List of number of images in each class
    
    for name in classes_names:
        nb_in_name = len(os.listdir(data_path + f'/{name}')) # Number of non images in the class 'name'
        nb_in_classes.append(nb_in_name)
    
    batch_size = int(sum(nb_in_classes) * ratio)

    # loading training data and rescaling it using ImageDataGenerator
    train_datagen = ImageDataGenerator(dtype = 'float32', rescale = 1./255.)
    train_generator = train_datagen.flow_from_directory(data_path,
                                                    batch_size = batch_size,
                                                    target_size = image_shape[:2],
                                                    class_mode = 'categorical')

    # Create x and classes lists to store batches np.arrays for mean of the selected channel and for image classes
    classes = []
    x = []


    try:
        # Iterate over the generator
        for images, labels in train_generator:

            # Extract labels and images
            # images, labels = next(train_generator)

            # Mean of the second chanel for a batch of images
            batch_x = images[:, :, :, channel].reshape((images.shape[0], image_shape[0] * image_shape[1])).mean(axis = 1)
            x.append(batch_x)
            
            # Refactor labels into classes array
            batch_classes = np.apply_along_axis(lambda x: x[0], axis = 1, arr = labels)
            classes.append(batch_classes)

    except:
        print("The final batch was not processed :", images.shape, "\nBecause it is smaller than the current batch size :", batch_size)

    # Stack the results arrays for all batches
    classes_arr = np.hstack(classes)
    x_arr = np.hstack(x)

    # Plot
    sns.histplot(x = x_arr, hue = classes_arr, kde = True)
    plt.show()

