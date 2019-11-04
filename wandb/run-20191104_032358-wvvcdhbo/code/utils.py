import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn
import time
import os
import tqdm as tqdm

def get_img(fname, folder="input/train_images_525/train_images_525"):
    img = cv2.imread(os.path.join(folder, fname))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_mask(fname, mask_dir="input/mask"):
    return np.load(os.path.join(mask_dir, fname+'.npy'))

def mask_to_np(df, out):
    ids = df.im_id.values
    for id in tqdm(ids):
        mask = make_mask(df, image_name)
        np.save(mask, os.path.join(id, out))

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

def rle_decode(mask_rle: str = "", shape: tuple = (1400, 2100)):
    """Source: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools"""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")

def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1400, 2100)):
    """Source: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools"""
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
    return masks
