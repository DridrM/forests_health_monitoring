# Basic imports
import numpy as np
import matplotlib.pyplot as plt

# Special imports for image preprocessing
import cv2
import PIL
import os
# Read pickle files (vector shapes)
import pickle
import shapely
import pandas as pd
import shutil

def open_pickle_polygons(pickle_path: str, read_option = '+rb') -> list:
    """Open a shapely object saved into a pickle file
       Transform the multipolygon object into a list of polygon objects"""

    with open(pickle_path, read_option) as f:
        polygons = pickle.load(f)

    try:
        myList=list(polygons.geoms)
    except:
        myList=[polygons]
    return myList

def apply_mask_to_image(image: np.array,
                        polygons: list,
                        alpha: float,
                        mask_color = (255, 255, 255),
                        line_type = cv2.LINE_4) -> np.array:
    """Apply a list of polygonal masks to an image"""

    # Function to round coordinates of the exterior of the polygon
    int_coords = lambda x: np.array(x).round().astype(np.int32)

    # Make a copy we can modify
    image_to_mask = image.copy()

    # Iterate over the polygons
    for polygon in polygons:

        # Round the coordinates of polygon
        exterior = [int_coords(polygon.exterior.coords)]

        # Create an overlay image
        overlay = image_to_mask.copy()

        # Create the mask on the overlay
        cv2.fillPoly(overlay, exterior, color = mask_color, lineType = line_type)



        # Flatten the mask and the image layer
        cv2.addWeighted(overlay, alpha, image_to_mask, 1 - alpha, 0, image_to_mask)

    return image_to_mask


def processMask(data_dir,alpha,out_name):
    all_files = os.listdir(data_dir)

    # Filter the list to include only files with a .png extension
    my_file = [file for file in all_files if file.endswith('.png')]

    #read pcikle file
    pickle_file= data_dir+"/forest_loss_region.pkl"
    print(pickle_file)
    mypoligons= open_pickle_polygons(pickle_file)

    #img_path= data_dir+"/"+img_name
    img_path= data_dir+"/"+my_file[0]
    image = cv2.imread(img_path)

    myImg=apply_mask_to_image(image,mypoligons,alpha)

    out_img = data_dir+"/"+out_name
    cv2.imwrite(out_img, myImg)

def bulkProcess(path_to_images,alpha, out_name):
   # myDirList=os.listdir(path_to_images)
    myDirList = [file for file in os.listdir(path_to_images) if not file.startswith('.')]
    for data_dir_l in myDirList:
        data_dir=path_to_images+"/"+data_dir_l
        try:
            processMask(data_dir,alpha,out_name)
        except Exception as e:
            print(data_dir, " not processed:",e)

def classifyData(csv_path,processed_file):
    train = pd.read_csv(csv_path+'/train.csv')
    path_to_train="../raw_data/ForestNetDataset/train"

    for index, row in train.iterrows():
        row_label=row["merged_label"]
        row_path=csv_path+"/"+row["example_path"]
        processed_path=row_path+"/"+processed_file
        img_name=train['example_path'].str.extract(r'examples/(.*)')
        file_exist=os.path.exists(processed_path)

        if row_label=="Plantation":
            my_input=f"Plantation: {processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Plantation/"+img_name
        elif row_label.startswith("Grass"):
            my_input=f"Grass: {processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Grass/"+img_name

        elif row_label.startswith("Small"):
            my_input=f"Agriculture:{processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Agriculture/"+img_name

        elif row_label=="Other":
            my_input=f"Other:{processed_path} - {file_exist}"
            my_test1=path_to_train+"/Other"
            my_target_file=path_to_train+"/Other/"+img_name
        else:
            print(img_name+"not found")
        print(processed_path)
        print(my_target_file)
#        shutil.copy(processed_path, my_target_file)


def pullFile(file_path):
    myDirList = [file for file in os.listdir(file_path) if not file.startswith('.')]
    print("New1")
    for cur_dir in myDirList:
        my_img= [file for file in os.listdir(file_path+"/"+cur_dir+"/images/visible") if not file.startswith('.')]
        my_img.sort()
        my_im_f=my_img[0]
        if my_im_f=="composite.png":
            print(my_img[0])
            print(cur_dir)
        my_im_f_input=file_path+"/"+cur_dir+"/images/visible/"+my_im_f
        my_target=file_path+"/"+cur_dir+"/"+my_im_f
        print(my_im_f_input)
        print(my_target)
        shutil.copy(my_im_f_input, my_target)
