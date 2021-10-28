import os

import pandas as pd
import random
from coco_converter import convert_coco
from utils.project_utils import write_json

data_dir = "../data/sartorius-cell-instance-segmentation"

# Dataset directories
train_dir = os.path.join(data_dir, "train")
train_csv = os.path.join(data_dir, "train.csv")

targets = ['shsy5y', 'astro', 'cort']

# Train set: shsy5y 155 images, astro 131 images, cort 320 images
# val_sizes: number of images of each class in validation set (10%)
val_sizes = {'shsy5y': 15, 'astro': 13, 'cort': 32}

train_df = pd.read_csv(train_csv)
img_split_mapping = dict()

for cls in targets:
    img_list = train_df[train_df['cell_type'] == cls]['id'].unique().tolist()
    random.shuffle(img_list)
    img_split_mapping[cls] = {'train': img_list[val_sizes[cls]:], 'val': img_list[:val_sizes[cls]]}

train_images = []
val_images = []

for cls, splits in img_split_mapping.items():
    train_images.extend(splits['train'])
    val_images.extend(splits['val'])

random.shuffle(train_images)
random.shuffle(val_images)

new_train_df = train_df[train_df['id'].isin(train_images)]
val_df = train_df[train_df['id'].isin(val_images)]

category_ids = {'shsy5y': 1, 'astro': 2, 'cort': 3}
train_json = convert_coco(new_train_df, category_ids)
train_path = os.path.join("../data", "train_annotations.json")
write_json(train_json, train_path)

val_json = convert_coco(val_df, category_ids)
val_path = os.path.join("../data", "val_annotations.json")
write_json(val_json, val_path)
