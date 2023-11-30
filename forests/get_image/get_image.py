## import package needed
import os
import shutil
from datetime import datetime
import pandas as pd

## get to the right directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

data_dir = os.path.join(project_path, 'raw_data', 'ForestNetDataset')

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
train['dataset'] = 'train'
# test =  pd.read_csv(os.path.join(data_dir, 'test.csv'))
#test['dataset'] = 'test'
# val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
#val['dataset'] = 'val'
# dataset = pd.concat([train, test, val], axis=0)




#########################################################################################################
            ## for loop to iterate on each file in 'example' and copy select file in each file
#########################################################################################################
folders = train['example_path'].tolist()
for f in folders:
    images_path = os.path.join(data_dir, f, 'images', 'visible')
    if os.path.isdir(images_path):
        images = os.listdir(images_path)
        images.sort()
        # shutil.copy(
        #     os.path.join(images_path, images[0]),
        #     os.path.join(data_dir, f)
        # )
        print(images[0])
