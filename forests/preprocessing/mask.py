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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image


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

def process_mask_target(input_pickle_path: str,
                        output_mask_target_path: str,
                        mask_size = (332, 332, 1)) -> None:
    """Build a mask from the shape at input_pickle_path
       Write the mask in the output_mask_target_path as numpy array"""

    # Create a 332 * 332 black image
    black_image = np.zeros(shape = mask_size)

    # Extract the polygons from the pickle
    polygons = open_pickle_polygons(input_pickle_path)

    # Apply mask to the black image
    target = apply_mask_to_image(black_image, polygons, alpha = 1, line_type = cv2.LINE_8)

    # Save the target mask
    np.save(output_mask_target_path, target)

def bulkProcess(path_to_images,alpha, out_name):
   # myDirList=os.listdir(path_to_images)
    myDirList = [file for file in os.listdir(path_to_images) if not file.startswith('.')]
    for data_dir_l in myDirList:
        data_dir=path_to_images+"/"+data_dir_l
        try:
            processMask(data_dir,alpha,out_name)
        except Exception as e:
            print(data_dir, " not processed:",e)

def classifyData(csv_path,target_path,processed_file, process_type):
    if process_type=="train":
        train = pd.read_csv(csv_path+'/train.csv')
        path_to_train=target_path+"/train"
    elif process_type=="valid":
        train = pd.read_csv(csv_path+'/val.csv')
        path_to_train=target_path+"/valid"
    elif process_type=="test":
        train = pd.read_csv(csv_path+'/test.csv')
        path_to_train=target_path+"/test"

    for index, row in train.iterrows():
        row_label=row["merged_label"]
        row_path=csv_path+row["example_path"]
        processed_path=row_path+"/"+processed_file
        image_str=row['example_path']
        img_name=image_str.replace('examples/',"")+".jpg"
        file_exist=os.path.exists(processed_path)

        if row_label=="Plantation":
           # my_input=f"Plantation: {processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Plantation/"+img_name
        elif row_label.startswith("Grass"):
            #my_input=f"Grass: {processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Grass/"+img_name

        elif row_label.startswith("Small"):
            #my_input=f"Agriculture:{processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Agriculture/"+img_name

        elif row_label=="Other":
            #my_input=f"Other:{processed_path} - {file_exist}"
            my_target_file=path_to_train+"/Other/"+img_name
        else:
            print(img_name+"not found")
        #print("PROCESSED PATH",processed_path)
        #print("TARGET PATH",my_target_file)
        try:
            shutil.copy(processed_path, my_target_file)
        except Exception as e:
            print(processed_path, " not processed:",e)



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


def process_mask_target(input_pickle_path: str, 
                        output_mask_target_path: str, 
                        mask_size = (332, 332, 1)) -> None:
    """Build a mask from the shape at input_pickle_path
       Write the mask in the output_mask_target_path as numpy array"""
       
    # Create a 332 * 332 black image
    black_image = np.zeros(shape = mask_size)
       
    # Extract the polygons from the pickle
    polygons = open_pickle_polygons(input_pickle_path)
    
    # Apply mask to the black image
    target = apply_mask_to_image(black_image, polygons, alpha = 1, line_type = cv2.LINE_8)
    
    # Save the target mask
    cv2.imwrite(output_mask_target_path, target)


def classifyData_Pickle(csv_path, process_type):
    
    pickle_name = "forest_loss_region.pkl"
    
    if process_type=="train":
        train = pd.read_csv(csv_path+'/train.csv')
        path_to_train="../raw_data/ForestNetDataset/train"
    
    elif process_type=="valid":
        train = pd.read_csv(csv_path+'/val.csv')
        path_to_train="../raw_data/ForestNetDataset/valid"
    
    elif process_type=="test":
        train = pd.read_csv(csv_path+'/test.csv')
        path_to_train="../raw_data/ForestNetDataset/test"

    for index, row in train.iterrows():
        row_path=csv_path+row["example_path"]
        all_files=os.listdir(row_path)
        png_files = [file for file in all_files if file.endswith('.png')]
        processed_path=row_path+"/"+png_files[0]
        image_str=row['example_path']
        target_dir=image_str.replace('examples/',"")
        img_name=target_dir+".png"
        
        input_pickle = row_path+"/"+pickle_name
        pickle_name_target=target_dir+"_mask.png"
        
        my_target_file=path_to_train + "/features/features" + img_name
        my_target_pickle=path_to_train + "/targets/targets" + pickle_name_target
        
        # print("INPUT IMAGE",processed_path)
        # print("INPUT PICKLE:",input_pickle)
        # print("TARGET IMAGE",my_target_file)
        # print("TARGET PICKLE",my_target_pickle)
        try:
             shutil.copy(processed_path, my_target_file)
            #  shutil.copy(input_pickle, my_target_pickle)
             process_mask_target(input_pickle, my_target_pickle)
             
        except Exception as e:
             print(processed_path, " not processed:", e)


def pltConfusionMatrix(model,test_ds):

    class_names=test_ds.class_names

    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)
    predicted_classes = np.argmax(y_pred_prob, axis=1)

    labels = test_ds.map(lambda x, y: y)
    y_test_n = np.array(list(labels))
    y_test=y_test_n[0,:,:]
    test_classes=np.argmax(y_test, axis=1)


    confusion_matrix_f = confusion_matrix(test_classes,predicted_classes, normalize='true' )*100
    sns.heatmap(confusion_matrix_f.T,annot=True);


def processMaskAsNumpy(data_dir):
    all_files = os.listdir(data_dir)

    # Filter the list to include only files with a .png extension
    my_file = [file for file in all_files if file.endswith('.png')]

    #read pcikle file
    pickle_file= data_dir+"/forest_loss_region.pkl"

    #img_path= data_dir+"/"+img_name
    img_path= data_dir+"/"+my_file[0]
    image = cv2.imread(img_path)
    target_img=image.copy()
    target_mask=process_mask_targetAsNumpy(pickle_file)
    target_img = np.dstack((image, target_mask))
    return target_img

def process_mask_targetAsNumpy(input_pickle_path: str,
                        mask_size = (332, 332, 1)) :
    """Build a mask from the shape at input_pickle_path
       Write the mask in the output_mask_target_path as numpy array"""

    # Create a 332 * 332 black image
    black_image = np.zeros(shape = mask_size)

    # Extract the polygons from the pickle
    polygons = open_pickle_polygons(input_pickle_path)

    # Apply mask to the black image
    target = apply_mask_to_image(black_image, polygons, alpha = 1, line_type = cv2.LINE_8)

    # Save the target mask
    return target


def cropImage(img_path,crop_fact):
    all_files = os.listdir(img_path)
    my_file = [file for file in all_files if file.endswith('.png')]
    img_name=my_file[0]
    # Open the image file
    image = Image.open(img_path+"/"+img_name)

    # Get the dimensions of the image
    width, height = image.size

    # Define the size of the center portion you want to extract (e.g., 100x100 pixels)
    center_width, center_height = round(width/2), round(height/2)

    # Calculate the coordinates for cropping the center portion
    left = (width - center_width) // crop_fact
    top = (height - center_height) // crop_fact
    right = (width + center_width) // crop_fact
    bottom = (height + center_height) // crop_fact

    # Crop the center portion of the image
    center_image = image.crop((left, top, right, bottom))

    # Save or display the cropped center portion
    crop_name=f"{img_path}/image_cropped_{crop_fact}.jpg"
    center_image.save(crop_name)
    #center_image.show()

def bulkProcessCrop(path_to_images,crop_fact):
    myDirList=os.listdir(path_to_images)
 #   print(myDirList)
    #myDirList = [file for file in os.listdir(path_to_images) if not file.startswith('.')]
    for data_dir_l in myDirList:
        data_dir=path_to_images+"/"+data_dir_l
        try:
            cropImage(data_dir,crop_fact)
        except Exception as e:
            print(data_dir, " not processed:",e)

