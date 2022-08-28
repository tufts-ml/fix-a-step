from os import listdir
import pandas as pd 
from itertools import product
import re
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
path_to_camus = '' # Fill with the path to the dataset
source_train = path_to_camus + 'raw/training/training/' # Put the original folders in a raw folder
source_test = path_to_camus + 'raw/testing/testing/'

dest_train = path_to_camus + 'std/training/' # Create a std folder
dest_test = path_to_camus + 'std/testing/'
train_paths = {"source path": source_train, "dest path": dest_train}
test_paths = {"source path": source_test, "dest path": dest_test}
paths = [train_paths, test_paths]

# where file paths and labels will go for all the images
file_labeling = path_to_camus + 'std/labels_per_image.csv'


    


image_info = pd.DataFrame(columns=['tiff_paths', 'view_labels', 'diagnosis_labels'])



for path in paths:
    files = listdir(path["source path"])
    for patients in files:
        for (view, end), im_id in zip(product(['2CH', '4CH'], ['ED', 'ES']), range(1,5)):
            try:   
                # Open image
                orig_path = path["source path"] + patients + "/" + \
                    patients + "_" + view + "_" + end + ".mhd"     
                itkimage = sitk.ReadImage(orig_path, imageIO="MetaImageIO")

                # Resize image
                (a, b, c) = itkimage.GetSpacing()
                (x, y, z) = itkimage.GetSize()
                arr = sitk.GetArrayFromImage(itkimage)[0]
                x_new = x
                y_new = round(y * b /a)
                im_resized = cv2.resize(arr, (x_new, y_new))

                # Find patient id
                id_pat = re.compile('\d\d\d\d')
                id_num = id_pat.search(str(patients)).group()
                id_num = str(int(id_num) + 450) if path["source path"] == source_test else id_num

                # Save image
                dest_path = path['dest path'] + id_num + 's2_' + str(im_id) + '.tiff'
                plt.imsave(dest_path, im_resized, cmap='gray')

                # Create entry in dataframe
                view = 'A2C' if view == '2CH' else 'A4C'
                image_info = image_info.append({'tiff_paths': dest_path, 'view_labels': view, 'end_type': end, 'diagnosis_labels': 'Not_Provided'}, ignore_index=True)
            except:
                continue
image_info = image_info.reset_index(drop=True)
image_info.to_csv(file_labeling, index=False)           
            
            

    