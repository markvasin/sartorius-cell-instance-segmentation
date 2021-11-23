import detectron2.data.transforms as T
from detectron2.data import DatasetMapper


def get_dataset_mapper(cfg):
    return DatasetMapper(cfg,
                         is_train=True,
                         augmentations=
                         [
                             # T.RandomBrightness(0.8, 1.2),
                             T.RandomRotation(angle=[-20, 20]),
                             # T.RandomLighting(0.7),
                             T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                             T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                             # T.RandomCrop("relative", (0.9, 0.9)) => can remove many cells on edge of images
                         ])
