import os
import pickle

import numpy as np

from utils.project_utils import get_data_path


def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_class(result):
    return max(((i, res.shape[0]) for i, res in enumerate(result[0])), key=lambda x: x[1])[0]


def get_masks(result, img_size=(520, 704)):
    bbox, segm = result

    # get class of the highest occurrence object since an image can only contains one type of cell
    cls_id = get_class(result)
    bbox = bbox[cls_id]
    segm = segm[cls_id]

    # get threshold and minimum number of pixel to filter results
    threshold = CLASS_THRESHOLDS[cls_id]
    min_pixel = MIN_PIXELS[cls_id]

    # format and filter prediction
    pred_masks = []
    for i in range(len(segm)):
        # box = bbox[i][:4] # not used
        score = bbox[i][4]
        mask = segm[i]
        if score >= threshold:
            pred_masks.append((mask, score))

    # sort predictions by score
    pred_masks.sort(key=lambda x: x[1], reverse=True)
    pred_masks = [mask for mask, score in pred_masks]

    # remove overlapping masks and masks that are very small
    final_res = []
    used = np.zeros(img_size, dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= min_pixel:  # skip predictions with small area
            used += mask
            final_res.append(rle_encode(mask))
    return final_res


CLASS_THRESHOLDS = {0: 0.45, 1: 0.35, 2: 0.35}
MIN_PIXELS = {0: 75, 1: 150, 2: 75}
result_file = os.path.join(get_data_path(), 'query_inst_res.pkl')
result = pickle.load(open(result_file, 'rb'))
masks = get_masks(result)
masks
