import numpy as np
import os
from PIL import Image


class Echo:
    def __init__(self, dataset_path, transform_fn=None):
        self.dataset = np.load(dataset_path, allow_pickle=True) #need to use HWC version of the data
        self.transform_fn = transform_fn
        
    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        
        image = Image.fromarray(image)
#         ###print the mode of the img to see if it is one of the (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
#         print('img mode is {}'.format(image.mode))
        
        if self.transform_fn is not None:
            image = self.transform_fn(image)
            
        return image, label

    def __len__(self):
        return len(self.dataset["images"])
    
    
    
    