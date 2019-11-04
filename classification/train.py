import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import numpy as np
import albumentations as albu
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader,Dataset
import cv2
from sklearn.model_selection import train_test_split
from utils import get_df, CloudDataset, seed_everything
from utils import get_training_augmentation, get_validation_augmentation
from utils import compute_pr_auc
from model import *
import configparser
from radam import RAdam
import wandb


seed_everything()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data_loaders(bs=8, num_workers=0, shuffle=True, ts=0.2):
    train_df, img_2_ohe_vector = get_df()
    train_imgs, val_imgs = train_test_split(train_df['Image'].values,
                            test_size=ts,
                            stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))),
                            random_state=42)
    print(train_imgs)
    print(val_imgs)
    print(len(train_imgs))
    print(len(val_imgs))
    train_dataset = CloudDataset(img_2_ohe_vector, img_ids=train_imgs,
                                 transforms=get_training_augmentation())
    train_loader = DataLoader(train_dataset, batch_size=bs,
                              shuffle=shuffle, num_workers=num_workers)

    val_dataset = CloudDataset(img_2_ohe_vector, img_ids=val_imgs,
                                 transforms=get_validation_augmentation())
    val_loader = DataLoader(val_dataset, batch_size=bs,
                              shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader


def train(num_epochs, model, optimizer, scheduler,
                    train_loader, val_loader, patience=3):
    """
    @param patience is for early stopping
    """
    prev_running_val_loss = 10000
    p_counter = 0 #When this counter is >= patience, stop training

    for epoch in tqdm(range(num_epochs)):
        print('=' * 12)
        print(f"Epoch {epoch}/{num_epochs-1}")
        print('=' * 12)

        running_train_loss = 0.
        running_train_auc = 0.
        running_val_loss = 0.
        running_val_auc = 0.

        # Train phase
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_auc = compute_pr_auc(labels, outputs)
                running_train_loss += loss.item()/len(train_loader)
                running_train_auc += train_auc/len(train_loader)
        # print(f"TRAIN>>> Loss: {running_train_loss} | AUC: {running_train_auc}")

        # Val phase
        for inputs, labels in tqdm(val_loader):
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(labels)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_auc = compute_pr_auc(labels, outputs)
                running_val_loss += val_loss.item()/len(val_loader)
                running_val_auc += val_auc/len(val_loader)
        # print(f"VAL>>> Loss: {running_val_loss} | AUC: {running_val_auc}")

        log_dict = {
                    "Avg Train loss": running_train_loss,
                    "Avg Train AUC": running_train_auc,
                    "Avg Val loss": running_val_loss,
                    "Avg Val AUC": running_val_auc,
                    }
        scheduler.step(running_val_loss)

        if running_val_loss <= prev_running_val_loss:
            p_counter = 0 #reset counter
        else:
            p_counter += 1

        if p_counter >= patience:
            #Early stop
            print(f"\nEarly stopping at epoch: {epoch}\n")
            break

        prev_running_val_loss = running_val_loss
        print(log_dict)
        wandb.log(log_dict)

    return model

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('configs/config.ini')
    conf = config['DEFAULT']

    lr1 = conf.getfloat('lr1')
    lr2 = conf.getfloat('lr2')
    bs = conf.getint('bs')
    num_epochs = conf.getint('epochs')
    patience = conf.getint('patience') #For early stopping


    train_loader, val_loader = get_data_loaders(bs=bs)
    model = Effnet(4).to(device)

    wandb.init(project="cloud-classification")
    wandb.watch(model)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr1)
    # optimizer = RAdam(model.parameters(), lr=lr1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                    patience=3)

    model = train(num_epochs, model, optimizer, scheduler,
                    train_loader, val_loader, patience)

    PATH = "../models/classification/freezed.pth"
    torch.save(model.state_dict(), PATH)

    model.unfreeze()
    optimizer = optim.Adam(model.parameters(), lr=lr2)
    # optimizer = RAdam(model.parameters(), lr=lr2)
    model = train(num_epochs, model, optimizer, scheduler,
                    train_loader, val_loader, patience)

    PATH = "../models/classification/unfreezed.pth"
    torch.save(model.state_dict(), PATH)
