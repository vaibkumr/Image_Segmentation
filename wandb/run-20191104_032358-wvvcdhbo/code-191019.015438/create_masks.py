import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as albu

def get_img(image_path):
    """Load image from disk"""
    img = cv2.imread(image_path)
    return img

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

if __name__ == "__main__":
    #Load train.csv to make mask
    train = pd.read_csv(f"input/train.csv")
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    dir_ip = 'input/train_images_525/train_images_525'
    dir_op_mask = 'input/mask'

    for d in [dir_op_mask]:
        if not os.path.exists(d):
            os.makedirs(d)

    tfms = albu.Compose([albu.Resize(350, 525)]) #To resize
    bar = tqdm(os.listdir(dir_ip), postfix={"file":"none"})

    for file in bar:
        bar.set_postfix(ordered_dict={"file":file})
        path = os.path.join(dir_ip, file)
        img = get_img(path)
        mask = make_mask(train, file)
        tfmed = tfms(image=img, mask=mask)
        img = tfmed['image']
        mask = tfmed['mask']
        np.save(os.path.join(dir_op_mask, file), mask)
