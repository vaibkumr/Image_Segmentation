use_wandb = True

import os
import cv2
import collections
import random
import time
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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
import pickle
if use_wandb:
    import wandb

# Catalyst is amazing.
from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
if use_wandb:
    from catalyst.dl.runner import SupervisedWandbRunner as SupervisedRunner
else:
    from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

# PyTorch made my work much much easier.
import segmentation_models_pytorch as smp
from dataloader import SegmentationDataset, SegmentationDatasetTest
from augmentations import get_training_augmentation, get_validation_augmentation, get_preprocessing

from metric import BCEDiceLoss, DiceLoss, dice
from utils import *

device=torch.device('cuda')
# device=torch.device('cpu')
def seed_everything(seed=42):
    """
    42 is the answer to everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

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
        return loaders

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.ini",
                        help="config file for training hyperparameters")
    parser.add_argument("--wandb", "-w", default="segmentation-phase2",
                        help="wandb project name")
    args = parser.parse_args()
    project = args.wandb
    config_file = args.config

    if use_wandb:
        wandb.init(project=project)

    config = configparser.ConfigParser()
    config.read(config_file)
    conf = config['DEFAULT']

    lrd = conf.getfloat('lrd') #decoder lr
    lre = conf.getfloat('lre') #encoder lr
    epochs = conf.getint('epochs')
    num_workers = conf.getint('num_workers')
    bs = conf.getint('bs')
    s_patience = conf.getint('s_patience')
    train_patience = conf.getint('train_patience')
    arch = conf.get('arch')
    encoder = conf.get('encoder')
    logdir = f"./logs/{arch}_{encoder}"

    model, preprocessing_fn = get_model(encoder, type=arch)
    if use_wandb:
        wandb.watch(model)
    loaders = get_loaders(bs, num_workers, preprocessing_fn)

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': lrd},
        {'params': model.encoder.parameters(), 'lr': lre},
    ])

    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.6, patience=s_patience)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    # criterion = BCEDiceLoss()
    # criterion = DiceLoss(eps=1.) #Try this too
    runner = SupervisedRunner()

    # Train
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(),
                   EarlyStoppingCallback(patience=train_patience,
                                        min_delta=0.001)
                   ],
        logdir=logdir,
        num_epochs=epochs,
        verbose=True
    )
    secs = time.time() - start
    print(f"Done in {secs:.2f} seconds ({secs/3600:.2f} hours)")
