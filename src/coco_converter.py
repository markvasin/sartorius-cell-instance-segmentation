from datetime import datetime

import numpy as np
from pycococreatortools import pycococreatortools as pc
from tqdm import tqdm


def rle_decode(mask_rle, shape=(520, 704), color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image image is actually flattened since RLE is a 1D "run"
    if len(shape) == 3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo: hi] = color

    # Don't forget to change the image back to the original shape
    return img.reshape(shape)


def convert_coco(train_df):
    cat_ids = {name: i + 1 for i, name in enumerate(train_df.cell_type.unique())}
    categories = [{'id': cls_id, 'name': cls_name} for cls_name, cls_id in cat_ids.items()]
    images = [
        {
            'id': img_id,
            'width': row.width,
            'height': row.height,
            'file_name': f'train/{img_id}.png'
        } for img_id, row in train_df.groupby('id').agg('first').iterrows()
    ]
    annotations = []

    for idx, row in tqdm(train_df.iterrows()):
        binary_mask = rle_decode(row.annotation, (row.height, row.width))
        category_info = {'id': cat_ids[row.cell_type], 'is_crowd': 0}
        annotation = pc.create_annotation_info(idx + 1, row.id, category_info, binary_mask, (row.width, row.height),
                                               tolerance=1)
        annotations.append(annotation)

    coco_json = {
        'images': images,
        'categories': categories,
        'annotations': annotations,
        'info': {
            'year': datetime.now().year,
            'version': '1.0',
            'contributor': 'Kaggle-Sartorius'
        }
    }
    return coco_json
