from os import listdir
import pandas as pd 
import matplotlib.pyplot as plt
import SimpleITK as sitk

path_to_unity = '' # Fill with path to unity set
source = path_to_unity + 'raw/images/png-cache/01/' # Put images in a raw folder
dest = path_to_unity + 'std/'
file_labeling = '' # Fill with desired path to comprehensive csv

image_info = pd.DataFrame(columns=['tiff_paths', 'view_labels', 'diagnosis_labels'])

# This file can be downloaded from https://data.unityimaging.net/files/view_index.csv
viewlabels_csv = '' # Fill with path to view labeled csv

labels = pd.read_csv(viewlabels_csv)

in_png = listdir(source)
id_num = 1

for first_layer in in_png:
    in_first_two_nums = listdir(source + first_layer)
    for second_layer in in_first_two_nums:
        in_second_two_nums = listdir(source + first_layer + '/' + second_layer)
        for patient_file in in_second_two_nums:
            
            # Open image
            orig_path = source + first_layer + '/' + second_layer + '/' + patient_file
            itkimage = sitk.ReadImage(orig_path)
            arr = sitk.GetArrayFromImage(itkimage)

            # Save image
            dest_path = dest + str(id_num) + 's2_' + '1.tiff' 
            id_num += 1
            plt.imsave(dest_path, arr, cmap='gray')

            # Save info in the dataframe 
            view = labels.loc[labels['file'] == patient_file].iloc[0]['view']
            view = view.upper()
            image_info = image_info.append({'tiff_paths': dest_path, 'view_labels': view, 'diagnosis_labels': 'Not_Provided'}, ignore_index=True)

image_info = image_info.reset_index(drop=True)
image_info.to_csv(file_labeling, index=False)           

            
            

    