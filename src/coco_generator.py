# From https://www.kaggle.com/coldfir3/coco-dataset-generator/notebook?scriptVersionId=78834016
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm


def rle2mask(rle, img_w, img_h):
    array = np.fromiter(rle.split(), dtype=np.uint)
    array = array.reshape((-1, 2)).T
    array[0] = array[0] - 1

    starts, lenghts = array

    mask_decompressed = np.concatenate([np.arange(s, s + l, dtype=np.uint) for s, l in zip(starts, lenghts)])

    msk_img = np.zeros(img_w * img_h, dtype=np.uint8)
    msk_img[mask_decompressed] = 1
    msk_img = msk_img.reshape((img_h, img_w))
    msk_img = np.asfortranarray(msk_img)

    return msk_img

# sartorius rle -> mask -> coco rle
def coco_structure(df):
    cat_ids = {name: id + 1 for id, name in enumerate(df.cell_type.unique())}
    cats = [{'name': name, 'id': id} for name, id in cat_ids.items()]
    images = [{'id': id, 'width': row.width, 'height': row.height, 'file_name': f'train/{id}.png'} for id, row in
              df.groupby('id').agg('first').iterrows()]
    annotations = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mask = rle2mask(row['annotation'], row['width'], row['height'])
        c_rle = maskUtils.encode(mask)
        c_rle['counts'] = c_rle['counts'].decode('utf-8')
        area = maskUtils.area(c_rle).item()
        bbox = maskUtils.toBbox(c_rle).astype(int).tolist()
        annotation = {
            'segmentation': c_rle,
            'bbox': bbox,
            'area': area,
            'image_id': row['id'],
            'category_id': cat_ids[row['cell_type']],
            'iscrowd': 0,
            'id': idx
        }
        annotations.append(annotation)
    return {'categories': cats, 'images': images, 'annotations': annotations}
