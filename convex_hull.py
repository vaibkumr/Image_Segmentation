import cv2
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

# helper functions
# credits: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools
class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
def draw_convex_hull(mask, mode='convex'):

    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        elif mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        elif mode=='approx':
            epsilon = 0.02*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255.

def rle_decode(mask_rle: str = '', shape = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')

def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(df, image_label, shape = (1400, 2100), cv_shape = (525, 350),debug=False):
    """
    Create mask based on df, image name and shape.
    """
    if debug:
        print(shape,cv_shape)
    df = df.set_index('Image_Label')
    encoded_mask = df.loc[image_label, 'EncodedPixels']
#     print('encode: ',encoded_mask[:10])
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded_mask is not np.nan:
        mask = rle_decode(encoded_mask,shape=shape) # original size

    return cv2.resize(mask, cv_shape)

min_size = [10000 ,10000, 10000, 10000]
def post_process_minsize(mask, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(mask.shape)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions #, num

# def show_image(image,figsize=None,title=None):
#
#     if figsize is not None:
#         fig = plt.figure(figsize=figsize)
# #     else: # crash!!
# #         fig = plt.figure()
#
#     if image.ndim == 2:
#         plt.imshow(image,cmap='gray')
#     else:
#         plt.imshow(image)
#
#     if title is not None:
#         plt.title(title)
#
# def show_Nimages(imgs,scale=1):
#
#     N=len(imgs)
#     fig = plt.figure(figsize=(25/scale, 16/scale))
#     for i, img in enumerate(imgs):
#         ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])
#         show_image(img)
#     plt.show()
#
# def draw_masks(img2,img_mask_list):
#
#     img = img2.copy()
#     for ii in range(4): # for each of the 4 masks
#         color_mask = np.zeros(img2.shape)
#         temp_mask = np.ones([img2.shape[0],img2.shape[1]])*127./255.
#         temp_mask[img_mask_list[ii] == 0] = 0
#         if ii < 3: # use different color for each mask
#             color_mask[:,:,ii] = temp_mask
#         else:
#             color_mask[:,:,0],color_mask[:,:,1],color_mask[:,:,2] = temp_mask,temp_mask,temp_mask # broadcasting to 3 channels
#
#         img += color_mask
#
#     return img
