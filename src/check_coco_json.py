import os
from collections import Counter

from src.utils.project_utils import get_data_path, read_json

train_path = os.path.join(get_data_path(), 'sartorius-mask-data', 'annotations_train.json')
val_path = os.path.join(get_data_path(), 'sartorius-mask-data', 'annotations_val.json')

train_json = read_json(train_path)
val_json = read_json(val_path)

train_images = [img_info['file_name'] for img_info in train_json['images']]
val_images = [img_info['file_name'] for img_info in val_json['images']]

cls_dict = {1: 'shsy5y', 2: 'astro', 3: 'cort'}

train_ann = [cls_dict[ann_info['category_id']] for ann_info in train_json['annotations']]
val_ann = [cls_dict[ann_info['category_id']] for ann_info in val_json['annotations']]

train_ann_id = set([ann_info['image_id'] for ann_info in train_json['annotations']])
train_images_id = set([img_info['id'] for img_info in train_json['images']])

train_count = Counter(train_ann)
val_count = Counter(val_ann)

print('duplicate images in train and val:', set(train_images) & set(val_images))

print('train images:', len(train_images))
print('images with empty annotations in training set:', len(train_images_id) - len(train_ann_id))
print('val images:', len(val_images))
print('val annotations:', len(val_ann))
print('train_ann_count', train_count)
print('val_ann_count', val_count)
