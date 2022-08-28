import numpy as np
import pandas as pd
import os
import json
from tqdm import trange
from PIL import Image, ImageOps



from shared_utilities import make_dir_if_not_exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_save_dir',  help='where to save the tfrecords')
parser.add_argument('--ImageList_fullpaths', help='the extracted image_list')
parser.add_argument('--class_to_integer_mapping_dir', help='dir of the predefined mapping')
parser.add_argument('--resized_shape', default=112, help='resize')
parser.add_argument('--sanity_verbose', default=False, help='whether to print some debugging statements')




def resize_and_pad(tiff_image, resized_shape):
    
    this_tiff_image_original_height = tiff_image.size[1]
    this_tiff_image_original_width = tiff_image.size[0]
    
    if this_tiff_image_original_height > this_tiff_image_original_width:
        
        this_tiff_image_new_height = resized_shape
        this_tiff_image_new_width = int(resized_shape * (this_tiff_image_original_width/this_tiff_image_original_height))
        
        pad_along_width = resized_shape - this_tiff_image_new_width
        pad_along_height = 0
        
        pad_configuration = ((0, pad_along_height), (0, pad_along_width), (0,0)) 
    
    elif this_tiff_image_original_height <= this_tiff_image_original_width:
        
        this_tiff_image_new_width = resized_shape
        this_tiff_image_new_height = int(resized_shape * (this_tiff_image_original_height/this_tiff_image_original_width))
        
        pad_along_height = resized_shape - this_tiff_image_new_height
        pad_along_width = 0
        
        pad_configuration = ((0, pad_along_height), (0, pad_along_width), (0,0))
    
    return this_tiff_image_new_height, this_tiff_image_new_width, pad_configuration


def main(result_save_dir, ImageList_fullpaths, class_to_integer_mapping_dir, resized_shape, sanity_verbose):
    
    make_dir_if_not_exists(result_save_dir)
    
    with open(os.path.join(class_to_integer_mapping_dir, 'view_class_to_integer_mapping.json')) as view_file:
        view_class_to_integer_mapping = json.load(view_file)
        
    ImageList_fullpaths = pd.read_csv(ImageList_fullpaths)
    
    extracted_PLAX_image = []
    extracted_PLAX_viewlabel = []
    
    extracted_A4C_image = []
    extracted_A4C_viewlabel = []
    
    extracted_A2C_image = []
    extracted_A2C_viewlabel = []

    
    
    num_images_to_extract = len(ImageList_fullpaths)
    print('#images to extract: {}'.format(num_images_to_extract))
    
    for i in trange(num_images_to_extract):
        this_tiff_fullpath = ImageList_fullpaths.iloc[i].path

        this_tiff_viewlabel = ImageList_fullpaths.iloc[i].view_labels
          
        tiff_image = Image.open(this_tiff_fullpath)
          
        if sanity_verbose:
            print('Before converting to grayscale. shape: {}, value is of type: {}, min: {}, max: {}'.format(np.array(tiff_image).shape, np.array(tiff_image).dtype, np.min(np.array(tiff_image)), np.max(np.array(tiff_image))))
        
        #convert to grayscale
        tiff_image = ImageOps.grayscale(tiff_image) #convert to gray scale
        
        if sanity_verbose:
            print('After converting to grayscale. shape: {}, value is of type: {}, min: {}, max: {}'.format(np.array(tiff_image).shape, np.array(tiff_image).dtype, np.min(np.array(tiff_image)), np.max(np.array(tiff_image))))
        
        #resize
        this_tiff_image_new_height, this_tiff_image_new_width, pad_configuration = resize_and_pad(tiff_image, resized_shape)
        tiff_image = tiff_image.resize((this_tiff_image_new_width, this_tiff_image_new_height))

        #expand from (H,W) to (H,W,1)
        tiff_image_array = np.expand_dims(np.array(tiff_image), axis=2)

        #pad
        tiff_image_array = np.pad(tiff_image_array, pad_width=pad_configuration, mode='constant', constant_values=0)
        tiff_image_array = np.stack([tiff_image_array.squeeze(), tiff_image_array.squeeze(), tiff_image_array.squeeze()], axis=-1)
        print('after stacking, tiff_image_array shape: {}'.format(tiff_image_array.shape))
        
        #sanity check print statements:
        if sanity_verbose:
            print('After resizing, shape: {}, value is of type: {}, min: {}, max: {}'.format(tiff_image_array.shape, tiff_image_array.dtype, np.min(tiff_image_array), np.max(tiff_image_array)))


        if this_tiff_viewlabel=='PLAX':
            extracted_PLAX_image.append(tiff_image_array)
            extracted_PLAX_viewlabel.append(view_class_to_integer_mapping[this_tiff_viewlabel])
            
        elif this_tiff_viewlabel=='A4C':
            extracted_A4C_image.append(tiff_image_array)
            extracted_A4C_viewlabel.append(view_class_to_integer_mapping[this_tiff_viewlabel])           
        elif this_tiff_viewlabel=='A2C':
            extracted_A2C_image.append(tiff_image_array)
            extracted_A2C_viewlabel.append(view_class_to_integer_mapping[this_tiff_viewlabel]) 

    
    
    extracted_PLAX_image = np.array(extracted_PLAX_image)
    extracted_A4C_image = np.array(extracted_A4C_image)
    extracted_A2C_image = np.array(extracted_A2C_image)
    total_extracted_image = np.concatenate((extracted_PLAX_image,extracted_A2C_image,extracted_A4C_image), axis=0)
    total_extracted_viewlabel = np.concatenate((extracted_PLAX_viewlabel, extracted_A2C_viewlabel, extracted_A4C_viewlabel), axis=0)
    
    print('total_extracted_image: {}'.format(total_extracted_image.shape),flush=True)
    


    
    num_extracted_PLAX = len(extracted_PLAX_image)
    num_extracted_A4C = len(extracted_A4C_image)
    num_extracted_A2C = len(extracted_A2C_image)
    
    #sanity check:
    assert num_extracted_PLAX+num_extracted_A4C+num_extracted_A2C==num_images_to_extract

    #write to txt file:
    make_dir_if_not_exists(os.path.join(result_save_dir, 'stats/'))

    file_writer = open(os.path.join(result_save_dir, 'stats/', 'singlelabel_stats.txt'), 'w')
    
    file_writer.write('total images in this image list: {}\n'.format(num_images_to_extract))
    file_writer.write('num_extracted_PLAX: {}\n'.format(num_extracted_PLAX))
    file_writer.write('num_extracted_A4C: {}\n'.format(num_extracted_A4C))
    file_writer.write('num_extracted_A2C: {}\n'.format(num_extracted_A2C))
       
    file_writer.close()
    
    with open(os.path.join(result_save_dir, 'train_label.npy'), 'wb') as f:
        np.save(f, total_extracted_viewlabel)

    with open(os.path.join(result_save_dir, 'train_.npy'), 'wb') as f:
        np.save(f, total_extracted_image)  

    with open(os.path.join(result_save_dir, 'train_PLAX_image.npy'), 'wb') as f:
        np.save(f, extracted_PLAX_image)
    
    with open(os.path.join(result_save_dir, 'train_PLAX_label.npy'), 'wb') as f:
        np.save(f, extracted_PLAX_viewlabel)
        
    with open(os.path.join(result_save_dir, 'train_A4C_image.npy'), 'wb') as f:
        np.save(f, extracted_A4C_image)
    
    with open(os.path.join(result_save_dir, 'train_A4C_label.npy'), 'wb') as f:
        np.save(f, extracted_A4C_viewlabel)
    
    with open(os.path.join(result_save_dir, 'train_A2C_image.npy'), 'wb') as f:
        np.save(f, extracted_A2C_image)
    
    with open(os.path.join(result_save_dir, 'train_A2C_label.npy'), 'wb') as f:
        np.save(f, extracted_A2C_viewlabel)

        
    

    
    
if __name__=="__main__":
    
    args = parser.parse_args()
    
    result_save_dir = args.result_save_dir
    ImageList_fullpaths = args.ImageList_fullpaths
    class_to_integer_mapping_dir = args.class_to_integer_mapping_dir
    resized_shape = int(args.resized_shape)
    sanity_verbose = args.sanity_verbose
    
    main(result_save_dir, ImageList_fullpaths, class_to_integer_mapping_dir, resized_shape, sanity_verbose)
