import tensorflow as tf
from tensorflow.keras import optimizers, regularizers,metrics
from tensorflow.keras.utils import load_img, img_to_array,image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    if num_classes >2:
        final_activation="softmax"
    else:
        final_activation="sigmoid"
    model = Sequential()
    model.add(Rescaling(1./255, input_shape = image_shape))
    model.add(Conv2D(filters = 32, kernel_size = kernel_size, input_shape = image_shape, activation="relu", padding = "same"))
    model.add(Conv2D(filters = 32, kernel_size = kernel_size, activation="relu", padding = "same"))
    model.add(Flatten())

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation=final_activation))

    return model

def createModelwithPooling(image_shape,kernel_size,num_classes):
    # use num_classes to specify the number of categories
    if num_classes >2:
        final_activation="softmax"
    else:
        final_activation="sigmoid"
    model = Sequential()
    model.add(Rescaling(1./255, input_shape = image_shape))
    model.add(Conv2D(filters = 64, kernel_size = kernel_size, input_shape = image_shape, activation="relu", padding = "same"))
    model.add(MaxPooling2D(pool_size=kernel_size))
    model.add(Conv2D(filters = 32, kernel_size = kernel_size, activation="relu", padding = "same"))
    model.add(MaxPooling2D(pool_size=kernel_size))
    model.add(Flatten())

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation=final_activation))

    return model

def compileModel(model, loss_type,learning_rate = 0.001):
    #  loss_type= ='binary_crossentropy' for 2 categories
    # loss_type= = 'categorical_crossentropy' for multiple
    adam = optimizers.Adam(learning_rate)
    model.compile(loss=loss_type,
              optimizer= adam,
              metrics=['accuracy','Recall'])
    return model

def trainModel(model,train_ds,val_ds,n_epochs=20):
    print("New model")
  #  EarlyStopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    history = model.fit(
            train_ds,
            epochs=n_epochs,
            validation_data=val_ds)
  #          callbacks = [EarlyStopper])

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


def createDataSet_ForestNet(data_path,image_size,batch_size,label_mode,test_sample_size,sampleEnv=True):

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
    batch_size=test_sample_size)

    return train_ds, val_ds , test_ds


def randomCrop_ForestNet(data_path,image_size,batch_size):

    # Define the parameters for your image dataset
 #   image_size = (224, 224)  # Specify the desired image size
 #   data_dir = 'path_to_your_dataset_directory'
    # USE LABEL MODE categorical for multiple categories
    train_path = data_path+"train"
    valid_path = data_path+"valid"
    test_path = data_path+"test"
    # Define data augmentation including random cropping
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomCrop(height=image_size[0], width=image_size[1])
    ])

    # Create the image dataset with random cropping
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        subset="training",
        labels="inferred",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",  # Or "binary" for binary classification
        shuffle=True,
        data_augmentation=data_augmentation  # Use data augmentation
    )

    # You can also create a validation dataset in a similar way
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        valid_path,
        subset="validation",
        labels="inferred",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False  # No need to shuffle the validation dataset
    )

def createBaseModel(image_size):
    base_model = tf.keras.applications.Xception(
         weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=image_size,
        include_top=False)  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False
    inputs = tf.keras.Input(shape=image_size)
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
    x = base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def showConfusion_Matrix(test_ds,model):
    y_pred_prob = model.predict(test_ds)
    labels = test_ds.map(lambda x, y: y)
    y_test_n = np.array(list(labels))
    y_test=y_test_n[0,:,:]
    test_classes=np.argmax(y_test, axis=1)
    predicted_classes = np.argmax(y_pred_prob, axis=1)
    confusion_matrix_f = confusion_matrix(test_classes,predicted_classes)
    disp=ConfusionMatrixDisplay(confusion_matrix_f,display_labels=test_ds.class_names)
    disp.plot()
