from __future__ import print_function, division
import os
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2

class LidcDataLoader(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
    Output:
        image = Transformed images
        mask = Transformed labels"""
    def __init__(self, images_dir, labels_dir):
        self.image_paths = images_dir
        self.label_paths= labels_dir
        self.augmentations = albu.Compose([
            albu.Rotate(limit=30, p=0.5),  # Rotate by up to 30 degrees
            albu.HorizontalFlip(p=0.15), # Horizontal flip
            albu.ElasticTransform(alpha=1.1, alpha_affine=0.5, sigma=5, p=0.5),  # Elastic transform
            #albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),  # Shift, scale, and rotate
            #albu.Affine(shear=[-15,15], translate_px=(-15, 15),p=0.5),
            ToTensorV2()
        ])
        self.transformations = transforms.Compose([transforms.ToTensor()])
       

    def transform(self, image, mask):
        #It is always best to convert the make input to 3 dimensional for albumentation
        image = image.reshape(512,512,1)
        mask = mask.reshape(512,512,1)
        # Without this conversion of datatype, it results in cv2 error. Seems like a bug
        mask = mask.astype('uint8')
        augmented=  self.augmentations(image=image,mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        mask= mask.reshape([1,512,512])

        image,mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image,mask
    
    def transform_no_aug(self, image, mask):
        image = self.transformations(image)
        mask = self.transformations(mask)
        image,mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image,mask

    def __getitem__(self, i):
        image = np.load(self.image_paths[i])
        mask = np.load(self.label_paths[i])
        image, mask = self.transform(image, mask)
        return image, mask
        

    def __len__(self):
        return len(self.image_paths)