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
import pickle


def get_ids(train_ids_file='../train_ids.pkl', valid_ids_file='../valid_ids.pkl'):
    with open(train_ids_file, 'rb') as handle:
        train_ids = pickle.load(handle)

    with open(valid_ids_file, 'rb') as handle:
        valid_ids = pickle.load(handle)

    return train_ids, valid_ids

def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    if img1.sum() + img2.sum() == 0:
        print("ok...")
        return 1

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


config = configparser.ConfigParser()
config.read('configs/config.ini')
conf = config['DEFAULT']
arch = conf.get('arch')
encoder = conf.get('encoder')

logdir = f"./logs/{arch}_{encoder}"
model_path = f"{logdir}/checkpoints/best.pth"
output_name = f"{logdir}/{arch}_{encoder}"

train_ids, valid_ids = get_ids()

sigmoid = lambda x: 1 / (1 + np.exp(-x))


bs = 8
num_workers = 0
# encoder = 'efficientnet-b4'
# arch = 'linknet'
model, preprocessing_fn = get_model(encoder, type=arch)
loaders = get_loaders(bs, num_workers, preprocessing_fn)

valid_dataset = SegmentationDataset(ids=valid_ids,
                    transforms=get_validation_augmentation(),
                    preprocessing=get_preprocessing(preprocessing_fn),
                    img_db="../input/train_images_480/",
                    mask_db="../input/train_masks_480/", npy=True)

valid_loader = DataLoader(valid_dataset, batch_size=bs,
            shuffle=False, num_workers=num_workers)


runner = SupervisedRunner()
encoded_pixels = []
loaders = {"infer": valid_loader}
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[
        CheckpointCallback(
            resume=model_path),
        InferCallback()
    ],
)

import tqdm

valid_masks = []
probabilities = np.zeros((int(555*4), 350, 525))
for i, (batch, output) in enumerate(tqdm.tqdm(zip(
        valid_dataset, runner.callbacks[0].predictions["logits"]))):
    image, mask = batch
    for m in mask:
        if m.shape != (350, 525):
            m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        valid_masks.append(m)

    for j, probability in enumerate(output):
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        probabilities[i * 4 + j, :, :] = probability

class_params = {}
for class_id in range(4):
    print(class_id)
    attempts = []
    for t in tqdm.tqdm(range(0, 100, 5)):
        t /= 100
        for ms in [100, 5000, 10000, 15000, 20000, 22000, 25000]:
            masks = []
            for i in range(class_id, len(probabilities), 4):
                probability = probabilities[i]
                predict, num_predict = post_process(sigmoid(probability), t, ms)
                masks.append(predict)

            d = []
            for i, j in zip(masks, valid_masks[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))

            attempts.append((t, ms, np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])


    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    class_params[class_id] = (best_threshold, best_size)

with open(output_name+".pkl", 'wb') as handle:
    pickle.dump(class_params, handle)

import gc
torch.cuda.empty_cache()
gc.collect()

sub = pd.read_csv(f'../input/sample_submission.csv')
sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values


test_dataset = SegmentationDatasetTest(test_ids,
                                        transforms=get_test_augmentation(),
                                        preprocessing=get_preprocessing(preprocessing_fn),
                                        img_db="../input/test_images_525/test_images_525")

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
loaders = {"test": test_loader}

encoded_pixels = []
image_id = 0
for i, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
    runner_out = runner.predict_batch({"features": test_batch.cuda()})['logits']
    for i, batch in enumerate(runner_out):
        for probability in batch:

            probability = probability.cpu().detach().numpy()
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1

sub['EncodedPixels'] = encoded_pixels
# Classifer use

import pickle
with open('Data/list.pkl', 'rb') as handle:
    image_labels_empty = pickle.load(handle)
predictions_nonempty = set(submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values)
print(f'{len(image_labels_empty.intersection(predictions_nonempty))} masks would be removed')



sub.to_csv(output_name+".csv", columns=['Image_Label', 'EncodedPixels'], index=False)
