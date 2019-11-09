load_params = True

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

        train_dataset = SegmentationDataset(ids=train_ids,
                    transforms=get_training_augmentation(),
                    preprocessing=get_preprocessing(preprocessing_fn),
                    img_db=img_db,
                    mask_db=mask_db, npy=npy)
        valid_dataset = SegmentationDataset(ids=valid_ids,
                    transforms=get_validation_augmentation(),
                    preprocessing=get_preprocessing(preprocessing_fn),
                    img_db=img_db,
                    mask_db=mask_db, npy=npy)

        train_loader = DataLoader(train_dataset, batch_size=bs,
            shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=bs,
            shuffle=False, num_workers=num_workers)

        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }
        return valid_dataset, loaders

config = configparser.ConfigParser()
config.read('configs/config.ini')
conf = config['DEFAULT']
arch = conf.get('arch')
encoder = conf.get('encoder')

logdir = f"./logs/{arch}_{encoder}"
model_path = f"{logdir}/checkpoints/best.pth"
output_name = f"{logdir}/{arch}_{encoder}" #will be .pkl and .csv later

train_ids, valid_ids = get_ids()

sigmoid = lambda x: 1 / (1 + np.exp(-x))


bs = 4
num_workers = 0
# encoder = 'efficientnet-b4'
# arch = 'linknet'
model, preprocessing_fn = get_model(encoder, type=arch)

valid_dataset, loaders = get_loaders(bs, num_workers, preprocessing_fn)

train_loader = loaders['train']
valid_loader = loaders['valid']

print("Loading model")
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
loaders['train'] = train_loader
loaders['valid'] = valid_loader




size = (320, 480)
if load_params:
    print(">>>> Loading params")
    with open(output_name+"_params.pkl", 'rb') as handle:
        class_params = pickle.load(handle)
else:
    print("Learning threshold and min area")
    valid_masks = []
    LIMIT = 800
    probabilities = np.zeros((int(LIMIT*4), 320, 480)) #HARDCODED FOR NOW
    for i, (batch, output) in enumerate(tqdm.tqdm(zip(valid_dataset, runner.callbacks[0].predictions["logits"]))):
        if i >= LIMIT:
            break
        image, mask = batch
        for m in mask:
            # if m.shape != (350, 525):
            #     m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            # if probability.shape != (350, 525):
            #     probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability
    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in tqdm.tqdm(range(20, 100, 5)):
            t /= 100
            for ms in [5000, 10000, 15000, 20000, 25000, 27000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t,
                                                            ms, size=size)
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

    del probabilities

    with open(output_name+"_params.pkl", 'wb') as handle:
        pickle.dump(class_params, handle)

# Calculate train/valid dice
diceScore = {}
for phase in ['train', 'valid']:
    running_dice = 0
    image_id = 0
    for i, test_batch in enumerate(tqdm.tqdm(loaders[phase])):
        images, masks = test_batch
        runner_out = runner.predict_batch({"features": images.cuda()})['logits']
        for i, (mask, batch) in enumerate(zip(masks, runner_out)):
            for j, probability in enumerate(batch):
                probability = probability.cpu().detach().numpy()
                # if probability.shape != (350, 525):
                #     probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability),
                            class_params[image_id % 4][0], class_params[image_id % 4][1], size=size)
                running_dice += dice(predict, mask[j,:,:])
                image_id += 1
    diceScore[phase] = running_dice/image_id

print(f"\n\nDicescore: {diceScore}\n\n")
with open(f"{logdir}/train_test_loss.txt", 'w') as handle:
    text = f"Train: {diceScore['train']}\nTest: {diceScore['valid']}"
    handle.write(text)
