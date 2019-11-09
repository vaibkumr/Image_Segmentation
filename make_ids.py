import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

def make_ids(csv_file="input/train.csv"):
    train = pd.read_csv(csv_file)
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
    reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values,
                random_state=42, stratify=id_mask_count['count'], test_size=0.25) #1:3 split
    return train_ids, valid_ids


train_ids, valid_ids = make_ids()

print(f"Train len: {len(train_ids)}")
print(f"Valid len: {len(valid_ids)}")

with open("train_ids.pkl", "wb") as handle:
    pickle.dump(train_ids, handle)

with open("valid_ids.pkl", "wb") as handle:
    pickle.dump(valid_ids, handle)
