## import package
from forests.preprocessing import mask
import os
import shutil
import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

## set a directory to forest_health_monitoring
#root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
## set a directory to examples
#path_to_examples = root + '/raw_data/ForestNetDataset/examples/'

## transform the multipolygon mask into np.array with black background and white mask
def mask_into_npy(path_to_examples : str):

    """Set a directory input_pickle path to get the mask and set a directory output_mask_target_path where
    to copy the mask in the form of npy file in order to use the function process_mask_target in mask"""

    for file in os.listdir(path_to_examples):
        if file != '.DS_Store' and file.endswith('.npy') == False:
            input_pickle_path = os.path.join(path_to_examples,file, 'forest_loss_region.pkl')
            output_mask_target_path = os.path.join(path_to_examples, file,f'{file}.npy')
            mask.process_mask_target(input_pickle_path, output_mask_target_path)




def directory_mask_npy(root : str, ## directory to forests_health_monitoring
                csv_path : str, ## directory to ForestNetDataset
                 processed_file : str, ##path to the mask.npy file
                 process_type : str ##train, test or valid
                 ):

    """function to copy mask.npy into the right directory ex : raw_data/ForestNetDataset/train/Plantation"""

    if process_type=="train":
        train = pd.read_csv(csv_path+'/train.csv')
        path_to_train=os.path.join(root, "raw_data/ForestNetDataset/train")
    elif process_type=="valid":
        train = pd.read_csv(csv_path+'/val.csv')
        path_to_train=os.path.join(root, "raw_data/ForestNetDataset/valid")
    elif process_type=="test":
        train = pd.read_csv(csv_path+'/test.csv')
        path_to_train=os.path.join(root, "raw_data/ForestNetDataset/test")

    for index, row in train.iterrows():
        row_label=row["merged_label"]
        row_path=csv_path+row["example_path"]
        processed_path=os.path.join(row_path, processed_file)
        file_exist=os.path.exists(processed_path)

        if row_label=="Plantation":
            my_target_file=os.path.join(path_to_train, row_label, processed_file)

        elif row_label.startswith("Grass"):
            my_target_file=path_to_train+"/Grass/"+processed_file

        elif row_label.startswith("Small"):
            my_target_file=path_to_train+"/Agriculture/"+processed_file

        elif row_label=="Other":
            my_target_file=path_to_train+"/Other/"+processed_file
        else:
            print(processed_file+"not found")

        try:
            shutil.copy(processed_path, my_target_file)
        except:
            print()




def move_masknpy_to_directory(path_to_examples : str, ## path to file examples
                              csv_path : str ##
                              ):
    """move each mask.npy file into the right directory thanks to the function directory_mask_npy"""
    for file in os.listdir(path_to_examples):
        if file != '.DS_Store':
            ## process == the name off .npy which is the coordinate (same as each file in example file)
            process_file = f'{file}.npy'
            for groups in ['test', 'train', 'valid']:
                directory_mask_npy(csv_path,process_file, groups)




def add_canal_to_masknpy(csv_path):

    """for each jpg image and masknpy having the same name stacked it to add canal of masknpy to jpg (transform into array)"""

    type_data = ['test', 'train', 'valid']
    type_class = ['Agriculture', 'Plantation', 'Other', 'Grass']
    for t_data in type_data:
        for t_class in (type_class):
            path_im_npy = os.path.join(csv_path, t_data, t_class)
            for file in os.listdir(path_im_npy):
                ## set condition to work only with file which have the same name to stack image with corresponding mask
                if os.path.splitext(f'{file}.jpg')[0] == os.path.splitext(f'{file}.npy')[0]:
                    if file.endswith('.jpg'):
                        ## transform each jpg into np.array
                        image_array = Image.open(os.path.join(path_im_npy,file))
                        image_array = np.array(image_array)
                    else :
                        ## load npy file
                        nw_channel = np.load(os.path.join(path_im_npy,file))
                ## stack array image with array mask
                image_4_channel = np.dstack((image_array, nw_channel))
                ## reform image
                rgbn_image = Image.fromarray((image_4_channel * 255).astype(np.uint8))
                ## set the path and extension of the 'image'
                rgbn_image_path = os.path.join(path_im_npy, os.path.splitext(file)[0] + '_rgbn.npy')
                ## save it into numpy type file
                np.save(rgbn_image_path, rgbn_image)





def create_list_of_rgbn_npy(csv_path : str):
    """create a list of train test and valid containing path to rgbn.npy"""
    glob_path = glob.glob(f'{csv_path}/*/*/*')
    train_ds = []
    test_ds = []
    valid_ds = []
    for file in glob_path:
        if 'train' in file and file.endswith('rgbn.npy') == True :
            train_ds.append(file)
        elif 'test' in file and file.endswith('rgbn.npy') == True:
            test_ds.append(file)
        elif 'valid' in file and file.endswith('rgbn.npy') == True:
            valid_ds.append(file)

    return train_ds, test_ds, valid_ds





def add_target_(dataset):
    name_target = []
    for file in dataset:
        if 'Agriculture' in file:
            name_target.append(0)
        elif 'Plantation' in file:
            name_target.append(1)
        elif 'Grass' in file:
            name_target.append(2)
        elif 'Other' in file:
            name_target.append(3)
    return name_target



def create_target(train_ds: list,
                  test_ds : list,
                  valid_ds : list):
    train_ds.sort()
    train_target = add_target_(train_ds)
    test_ds.sort()
    test_target = add_target_(test_ds)
    valid_ds.sort()
    valid_target = add_target_(valid_ds)

    ## reshape target
    train_target = np.array(train_target).reshape(-1,1)
    test_target = np.array(valid_target).reshape(-1,1)
    valid_target = np.array(test_target).reshape(-1,1)

    return train_target, test_target, valid_target



def load_rgbn_npy(train_ds : list,
                  test_ds : list,
                  valid_ds : list):

    """create a list for train, test and valid of rgbn np.array"""

    type_dataset = [train_ds, test_ds, valid_ds]
    train_ds_np = []
    test_ds_np = []
    valid_ds_np = []
    for list_dataset in type_dataset:
        for dataset in list_dataset:
            if 'train' in dataset:
                #print(dataset)
                train_ds_np.append(np.load(dataset))
            elif 'test' in dataset:
                test_ds_np.append(np.load(dataset))
            elif 'valid' in dataset:
                valid_ds_np.append(np.load(dataset))

    train_ds_np = np.array(train_ds_np)
    test_ds_np = np.array(test_ds_np)
    valid_ds_np = np.array(valid_ds_np)

    return train_ds_np, test_ds_np, valid_ds_np


def encode(train_target,
           test_target,
           valid_target):
    ohe = OneHotEncoder(sparse_output=False)
    target_train_ohe = ohe.fit_transform(train_target)
    target_test_ohe = ohe.transform(test_target)
    target_valid_ohe = ohe.transform(valid_target)

    return target_train_ohe, target_test_ohe, target_valid_ohe


def train_model_fourth_channels(model,
                                train_ds_np,
                                target_train_ohe,
                                valid_ds_np,
                                target_valid_ohe,
                                n_epochs=20,
                                es=False):
    cb = []
    if es == True:
        EarlyStopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0, restore_best_weights=True)
        cb = [EarlyStopper]

    history = model.fit(
            train_ds_np, target_train_ohe,
            epochs=n_epochs,
            validation_data=(valid_ds_np, target_valid_ohe),
            callbacks = cb)

    return model,history
