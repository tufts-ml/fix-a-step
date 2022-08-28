import numpy as np
import os
from PIL import Image


class CIFAR100:
    def __init__(self, dataset_path, transform_fn=None):
        self.dataset = np.load(dataset_path, allow_pickle=True).item() #need to use HWC version of the data
        self.transform_fn = transform_fn
        
    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        
        image = Image.fromarray(image)
        if self.transform_fn is not None:
            image = self.transform_fn(image)
            
        return image, label

    def __len__(self):
        return len(self.dataset["images"])
    
    
