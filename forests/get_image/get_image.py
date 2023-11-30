## import package needed
import os
import shutil
from datetime import datetime

## get to the right directory
os.getcwd()
os.chdir("..")
root = os.getcwd()
path = root + '/raw_data/ForestNetDataset/examples/'

#########################################################################################################
                            #creating a function in order to extract date
#########################################################################################################

def extract_date(im_name):
    # split im_name with '_' take element corresponding to date (here the first 3)
    date = im_name.split('_')[:3]
    # rejoin it to have get the date
    return '_'.join(date)

#########################################################################################################
            ## for loop to iterate on each file in 'example' and copy select file in each file
#########################################################################################################

for files in os.listdir(path):
    ## set the path where we want to copy our image
    output_dir = '/Users/malliageiger/code/malliaggr/forests_health_monitoring/raw_data/ForestNetDataset/test'#path +files
    ## set the path to access or visible image
    #get_image = output_dir +'/images/visible/'
    get_image = path +files + '/images/visible/'
    #print(get_image)
    ## set the path for use it as argument for if statement
    path_composite = get_image + 'composite.png'
    ## condition is if there is only one image and that image is composite copy this image to output path
    if len(os.listdir(get_image))== 1:
        shutil.copy(path_composite, output_dir)
        #print(os.listdir(get_image))
    else:
        ## drop image name composite.png
        for list_image in [os.listdir(get_image)]:
            if 'composite.png' in list_image:
                list_image.remove('composite.png')
                print(list_image)
                ## sort image to get the older one
                sorted_image = sorted(list_image, key=lambda x: datetime.strptime(extract_date(x), '%Y_%m_%d'))
                path_image = get_image + sorted_image[0]
                shutil.copy(path_image, output_dir)
                #print(path_image)

print(len(os.listdir(output_dir)))
