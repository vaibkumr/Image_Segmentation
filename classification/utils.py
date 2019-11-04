from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import albumentations as albu
import cv2
import os
import numpy as np
import torch
import random
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

def get_img(fname, folder="../input/train_images_525/train_images_525"):
    img = cv2.imread(os.path.join(folder, fname))
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_training_augmentation():
    """ I got this idea from kaggle. To return augmentations from a function
    makes the code look much cleaner and easier to handle."""
    train_transform = [
        albu.CLAHE(p=1),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
#         albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=0),
#         albu.HueSaturationValue(p=0.5),
        albu.GridDistortion(),
#         albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
#         albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    train_transform = [
        albu.CLAHE(p=1),
#         albu.Normalize()
    ]
    return albu.Compose(train_transform)

class CloudDataset(Dataset):
    def __init__(
        self,
        img_2_ohe_vector,
        img_ids = None,
        transforms = None,
        preprocessing = None,
    ):
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.img_2_ohe_vector = img_2_ohe_vector


    def __getitem__(self, idx):
        image_name = self.img_ids[idx]

        img = get_img(image_name)
        augmented = self.transforms(image=img)
        img = augmented["image"]

        if self.preprocessing:
            pp = self.preprocessing(image=img)
            img = pp['image']

        label = self.img_2_ohe_vector[image_name]
        label = torch.tensor(label,
                dtype=torch.float32, device=device).squeeze()
        img = torch.Tensor(img.transpose(2, 0, 1)).type(torch.float32)
        return img/255.0, label #Normalize by /255.0

    def __len__(self):
        return len(self.img_ids)


def get_df(csv_file='../input/train.csv'):
    """Ideas here are taken from general kaggle kernel codes.
        especially the idea of storing img_2_ohe_vector mappings. I love it.
    """
    train_df = pd.read_csv(csv_file)
    train_df = train_df[~train_df['EncodedPixels'].isnull()]
    train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
    train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
    classes = train_df['Class'].unique()
    train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
    #ohe mapping (one hot encoding)
    img_2_ohe_vector = {img:vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}
    return train_df, img_2_ohe_vector

def compute_pr_auc(y_true, y_pred, n_classes=4):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        pr_auc_mean = 0
        for class_i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            if np.isnan(np.sum(precision)) or np.isnan(np.sum(recall)):
                continue
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc/n_classes
        return pr_auc_mean
