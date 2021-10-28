import os

import pandas as pd

from coco_converter import convert_coco
from utils.project_utils import write_json

DATA_DIR = "../data/sartorius-cell-instance-segmentation"

# Dataset directories
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SEMI_DIR = os.path.join(DATA_DIR, "train_semi_supervised")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

category_ids = {'shsy5y': 1, 'astro': 2, 'cort': 3}
train_df = pd.read_csv(TRAIN_CSV)

coco_json = convert_coco(train_df, category_ids)
out_path = os.path.join("../data", "annotations.json")
write_json(coco_json, out_path)
