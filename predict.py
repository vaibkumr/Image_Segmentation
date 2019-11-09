import os
import cv2
import collections
import time
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
import configparser
import argparse
import wandb

# Catalyst is amazing.
from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
# from catalyst.dl.runner import SupervisedWandbRunner as SupervisedRunner
from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

# PyTorch made my work much much easier.
import segmentation_models_pytorch as smp
from dataloader import SegmentationDataset, SegmentationDatasetTest, SegmentationDataset_withid
from augmentations import get_training_augmentation, get_preprocessing
from augmentations import get_test_augmentation, get_validation_augmentation

from utils import *
from metric import dice
import pickle


def get_loaders(bs=32, num_workers=4, preprocessing_fn=None,
            img_db="input/train_images_480/", mask_db="input/train_masks_480/",
            npy=True):
        train_ids, valid_ids = get_ids()

        valid_ids = valid_ids[:100]

        valid_dataset = SegmentationDataset(ids=valid_ids,
                    transforms=get_validation_augmentation(),
                    preprocessing=get_preprocessing(preprocessing_fn),
                    img_db=img_db,
                    mask_db=mask_db, npy=npy)

        valid_loader = DataLoader(valid_dataset, batch_size=bs,
            shuffle=False, num_workers=num_workers)

        loaders = {
            "infer": valid_loader
        }
        return loaders


config = configparser.ConfigParser()
config.read('configs/config.ini')
conf = config['DEFAULT']
arch = conf.get('arch')
encoder = conf.get('encoder')

logdir = f"./logs/{arch}_{encoder}"
model_path = f"{logdir}/checkpoints/best.pth"
output_name = f"{logdir}/{arch}_{encoder}" #will be .pkl and .csv later

sigmoid = lambda x: 1 / (1 + np.exp(-x))

bs = 4 #FFFFFSSSS LARGE Bs just doesnt work.. always CUDA out of memory for me :/
num_workers = 0
# encoder = 'efficientnet-b4'
# arch = 'linknet'
model, preprocessing_fn = get_model(encoder, type=arch)

with open(output_name+"_params.pkl", 'rb') as handle:
    class_params = pickle.load(handle)

import gc
torch.cuda.empty_cache()
gc.collect()

sub = pd.read_csv(f'input/sample_submission.csv')
sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

# Load model, weird way in catalyst
loaders = get_loaders()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

runner = SupervisedRunner()
# runner.infer(
#     model=model,
#     loaders=loaders,
#     callbacks=[
#         CheckpointCallback(
#             resume=model_path),
#         InferCallback()
#     ],
# )

test_dataset = SegmentationDatasetTest(test_ids,
                                        transforms=get_test_augmentation(),
                                        preprocessing=get_preprocessing(preprocessing_fn),
                                        img_db="input/test_images_525/test_images_525")

test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                            num_workers=num_workers)

loaders = {"test": test_loader}

encoded_pixels = []
image_id = 0
size = (350, 525) #Required output size by kaggle
for i, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
    runner_out = runner.predict_batch({"features": test_batch.cuda()})['logits']
    for i, batch in enumerate(runner_out):
        for probability in batch:
            probability = probability.cpu().detach().numpy()
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(sigmoid(probability),
                                                class_params[image_id % 4][0],
                                                class_params[image_id % 4][1],
                                                size=size)
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1

sub['EncodedPixels'] = encoded_pixels

# Use classifer
import pickle
with open('list.pkl', 'rb') as handle:
    image_labels_empty = pickle.load(handle)

predictions_nonempty = set(sub.loc[~sub['EncodedPixels'].isnull(), 'Image_Label'].values)
print(f'{len(image_labels_empty.intersection(predictions_nonempty))} masks would be removed')

sub.loc[sub['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
sub.to_csv(output_name+".csv", columns=['Image_Label', 'EncodedPixels'], index=False)


# git fetch --all && git reset --hard origin/master
