import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import ImageFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import pandas as pd
from scipy import stats

def compute_convolution(input_image, kernel):
    # Parameters
    kernel = np.array(kernel)
    kernel_height, kernel_width = kernel.shape

    img = np.squeeze(input_image) # Removes dimensions of size 1
    img_height, img_width = img.shape

    output_image = []

    for x in range(img_height - kernel_height + 1):
        arr = []

        for y in range(img_width - kernel_width + 1):

            a = np.multiply(img[x: x + kernel_height, y: y + kernel_width],
                            kernel)
            arr.append(a.sum())

        output_image.append(arr)

    return output_image

def plot_convolution(img, kernel, activation=False):
    ''' The following printing function ease the visualization'''

    img = np.squeeze(img)
    output_img = compute_convolution(img, kernel)
    if activation:
        output_img = np.maximum(output_img, 0)

    plt.figure(figsize=(10, 5))

    ax1 = plt.subplot2grid((3,3),(0,0), rowspan=3)
    ax1.imshow(img, cmap='gray')
    ax1.title.set_text('Input image')

    ax2 = plt.subplot2grid((3,3),(1, 1))
    ax2.imshow(kernel, cmap='gray')
    ax2.title.set_text('Kernel')

    ax3 = plt.subplot2grid((3,3),(0, 2), rowspan=3)
    ax3.imshow(output_img, cmap='gray')
    ax3.title.set_text('Output image')

    for ax in [ax1, ax2, ax3]:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    plt.show()

    def geData(SampleEnv=True):
        data_path= "../raw_data"

        if SampleEnv:
            train_path = data_path+"/train-sample"
            valid_path = data_path+"/valid-sample"
            test_path = data_path+"/test-sample"
        else:
            train_path = data_path+"/train"
            valid_path = data_path+"/valid"
            test_path = data_path+"/test"

        image_shape = (350,350,3)
        N_CLASSES = 2
        #BATCH_SIZE = 256
        BATCH_SIZE = 32
        # loading training data and rescaling it using ImageDataGenerator
        train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
        train_generator = train_datagen.flow_from_directory(train_path,
                                                        batch_size = BATCH_SIZE,
                                                        target_size = (350,350),
                                                        class_mode = 'categorical')

        # loading validation data and rescaling it using ImageDataGenerator
        valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
        valid_generator = valid_datagen.flow_from_directory(valid_path,
                                                        batch_size = BATCH_SIZE,
                                                        target_size = (350,350),
                                                        class_mode = 'categorical')

        # loading test data and rescaling it using ImageDataGenerator
        test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
        test_generator = test_datagen.flow_from_directory(test_path,
                                                        batch_size = BATCH_SIZE,
                                                        target_size = (350,350),
                                                        class_mode = 'categorical')


def plotImg(train_generator):
    # Get a batch of images and labels from the generator
    images, labels = next(train_generator)
    print(images.shape)
    #print(labels)
    #print(len(labels))
    # Display the first image from the batch
    plt.imshow(images[5])
    plt.title(f"Class: {labels[5]}")
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

def analytics(train_generator):
    myMeanLst_0=[]
    myMeanLst_1=[]
    myStdLst_0=[]
    myStdLst_1=[]
    images, labels = next(train_generator)
    #for batch_images, batch_labels in train_generator:
    for i in range(len(labels)):
        green_img = images[i,:,:,1]
        label=labels[i,0]
        mean = np.mean(green_img)
        std = np.std(green_img)
        if label==0:
            myMeanLst_0.append(mean)
            myStdLst_0.append(std)
        else:
            myMeanLst_1.append(mean)
            myStdLst_1.append(std)
    # Perform a two-sample independent t-test
    t_statistic, p_value = stats.ttest_ind(myMeanLst_0, myMeanLst_1)

    # Print the results
    print(f"t-statistic Means: {t_statistic}")
    print(f"p-value Means: {p_value}")

    # Determine if the difference is statistically significant at a 95% confidence level (alpha = 0.05)
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. The two groups are statistically different.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the two groups.")


    # Perform a two-sample independent t-test
    t_statistic, p_value = stats.ttest_ind(myStdLst_0, myStdLst_1)

    # Print the results
    print(f"t-statistic Std: {t_statistic}")
    print(f"p-value Std: {p_value}")

    # Determine if the difference is statistically significant at a 95% confidence level (alpha = 0.05)
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. The two groups are statistically different.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the two groups.")

    return myMeanLst_0,myMeanLst_1,myStdLst_0,myStdLst_1

    def myTest():
        print("Hello")
