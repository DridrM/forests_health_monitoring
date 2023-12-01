## import package needed
import os
import shutil
from datetime import datetime
import pandas as pd
# # ## get to the right directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def copy_image():
   # # ## creating dataframe with train/test/val values
   data_dir = os.path.join(project_path, 'raw_data', 'ForestNetDataset')
   type_df = ['train', 'test', 'val']
   dataset = []
   for typedf, ind in zip(type_df, range(3)):
        typedf_ind = pd.read_csv(os.path.join(data_dir, f'{typedf}.csv'))
        typedf_ind['dataset'] = f'{typedf}'
        dataset.append(typedf_ind)

   dataset = pd.concat(dataset,axis=0)
   #########################################################################################################
               ## for loop to iterate on each file in 'example' and copy select file in each file
   #########################################################################################################
   folders = dataset['example_path'].tolist()
   for f in folders:
       images_path = os.path.join(data_dir, f, 'images', 'visible')
       if os.path.isdir(images_path):
           images = os.listdir(images_path)
           images.sort()
           shutil.copy(
               os.path.join(images_path, images[0]),
               os.path.join(data_dir, f)
           )


def remove_image():
    data_dir = os.path.join(project_path, 'raw_data', 'ForestNetDataset', 'examples')
    for files in os.listdir(data_dir):
        try:
            for image in os.listdir(os.path.join(data_dir,files)):
                if image.endswith('.png'):
                    os.remove(os.path.join(data_dir, files, image))
        except:
            print(os.path.join(data_dir, files, image))


if __name__ == "__main__":
   copy_image()
   remove_image()
