import tensorflow as tf
from tensorflow.keras import optimizers, regularizers,metrics
from tensorflow.keras.utils import load_img, img_to_array,image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Rescaling
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

def createDataSet(image_size,batch_size,sampleEnv=True):
    data_path= "../raw_data"
    if sampleEnv:
        train_path = data_path+"/train-sample"
        valid_path = data_path+"/valid-sample"
        test_path = data_path+"/test-sample"
    else:
        train_path = data_path+"/train"
        valid_path = data_path+"/valid"
        test_path = data_path+"/test"

    train_ds = image_dataset_from_directory(train_path,
                                        labels="inferred",
                                        label_mode="binary",
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        seed=123)

    val_ds = image_dataset_from_directory(
    valid_path,
    labels = "inferred",
    label_mode = "binary",
    seed=123,
    image_size=image_size,
    batch_size=batch_size)


    test_ds = image_dataset_from_directory(
    test_path,
    labels = "inferred",
    label_mode = "binary",
    seed=123,
    image_size=image_size,
    batch_size=batch_size)

    return train_ds, val_ds , test_ds


def createModel(image_shape,kernel_size,num_classes):
    # use num_classes to specify the number of categories
    model = Sequential()
    model.add(Rescaling(1./255, input_shape = image_shape))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = image_shape, activation="relu", padding = "same"))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation="relu", padding = "same"))
    model.add(Flatten())

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="sigmoid"))

    return model

def compileModel(model, loss_type, learning_rate = 0.001):
    #  loss_type= ='binary_crossentropy' for 2 categories
    # loss_type= = 'categorical_crossentropy' for multiple
    adam = optimizers.Adam(learning_rate)
    model.compile(loss=loss_type,
              optimizer= adam,
              metrics=['accuracy','recall'])
    return model

def trainModel(model,train_ds,val_ds,n_epochs=20):
    EarlyStopper = EarlyStopping(monitor='val_loss', patience=2, verbose=0, restore_best_weights=True)

    history = model.fit(
            train_ds,
            epochs=n_epochs,
            validation_data=val_ds,
            callbacks = [EarlyStopper])

    return model,history


def plot_history(history,batch_size,image_size,model_name,saveFig=False):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    title = f"BATCH SIZE={batch_size}, IMAGE SIZE={image_size}"
    fig.suptitle(title, fontsize=16)
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")

    ax[2].set_title('recall')
    ax[2].plot(history.epoch, history.history["recall"], label="Train Recall")
    ax[2].plot(history.epoch, history.history["val_recall"], label="Validation Recall")
    ax[0].legend()
    ax[1].legend()
    if saveFig:
        path_to_model=f"'../models/{model_name}.png"
        print(path_to_model)
        plt.savefig(path_to_model)


def show_img(train_ds):
    for images, labels in train_ds.take(1):
        # Select the first image from the batch
        image = images[0].numpy()
    pil_img = tf.keras.utils.array_to_img(image)
    plt.imshow(pil_img)


def show_results(model,test_ds):
    result = model.evaluate(test_ds)
    myres= f"Test Accuracy:{np.round(result[1],2)} - Test Recall:{np.round(result[2],2)}"
    print(myres)


def createDataSet_2(image_size,batch_size,sampleEnv=True):
    data_path= "../raw_data/ForestFireDataset/"

    train_path = data_path+"train"
    valid_path = data_path+"valid"
    test_path = data_path+"test"

    train_ds= image_dataset_from_directory(train_path,
                                        labels="inferred",
                                        label_mode="binary",
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        seed=123)


    # Create the validation dataset with the specified validation split
    val_ds = image_dataset_from_directory(
    valid_path,
    labels='inferred',
    label_mode='binary',  # or 'categorical' depending on your problem
    image_size=image_size,
    batch_size=batch_size,
    seed=123  # Set the same random seed for consistency
    )
    test_ds = image_dataset_from_directory(
    test_path,
    labels = "inferred",
    label_mode = "binary",
    seed=123,
    image_size=image_size,
    batch_size=batch_size)

    return train_ds, val_ds , test_ds


def createDataSet_ForestNet(image_size, label_mode, batch_size, sampleEnv = True,):
    data_path= "../raw_data/ForestNetDataset/"

    # USE LABEL MODE categorical for multiple categories
    train_path = data_path+"train"
    valid_path = data_path+"valid"
    test_path = data_path+"test"

    train_ds= image_dataset_from_directory(train_path,
                                        labels="inferred",
                                        label_mode=label_mode,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        seed=123)


    # Create the validation dataset with the specified validation split
    val_ds = image_dataset_from_directory(
    valid_path,
    labels='inferred',
    label_mode=label_mode,  # or 'categorical' depending on your problem
    image_size=image_size,
    batch_size=batch_size,
    seed=123  # Set the same random seed for consistency
    )
    test_ds = image_dataset_from_directory(
    test_path,
    labels = "inferred",
    label_mode = label_mode,
    seed=123,
    image_size=image_size,
    batch_size=batch_size)

    return train_ds, val_ds , test_ds

