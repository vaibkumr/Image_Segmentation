import os
import cv2
import time
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
from utils import get_img, get_mask

"""
DataLoader. I have preprocessed and saved masks as images. Also, resized and
saved the images. I used someone else's code to decode RLE into masks.
"""

class SegmentationDataset(Dataset):
    def __init__(self, ids, transforms, preprocessing=False,
            img_db="input/train_images_525/train_images_525",
            mask_db="input/mask"):
        self.ids = ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.img_db = img_db
        self.mask_db = mask_db

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = get_img(id, self.img_db)
        mask = get_mask(id, self.mask_db)
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            pre = self.preprocessing(image=image, mask=mask)
            image = pre['image']
            mask = pre['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return "Training dataset for Segmentation task. Returns [img, mask(s)]."

class SegmentationDatasetTest(Dataset):
    def __init__(self, ids, transforms, preprocessing=False):
        self.ids = ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        id = self.ids(idx)
        image = get_img(id, db="input/test_images_525/test_images_525")
        augmented = self.transform(image=image)
        image = augmented['image']
        if self.preprocessing:
            pre = self.preprocessing(image=image)
            image = pre['image']
        return np.transpose(image, [2, 0, 1])

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return "Testing dataset for Segmentation task. Returns [img]."
