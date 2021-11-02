import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode

import src.utils.project_utils as util
from src.augmentations import get_dataset_mapper

setup_logger()
data_dir = util.get_data_path()
image_dir = os.path.join(data_dir, 'sartorius-cell-instance-segmentation')
ann_dir = os.path.join(data_dir, 'train_annotations.json')
register_coco_instances('sartorius_train', {}, ann_dir, image_dir)
metadata = MetadataCatalog.get('sartorius_train')
train_ds = DatasetCatalog.get('sartorius_train')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATALOADER.NUM_WORKERS = 0
train_dict = {ds['file_name']: ds for ds in train_ds}

train_loader = build_detection_train_loader(cfg, mapper=get_dataset_mapper(cfg))

for train_image_batch in train_loader:
    for idx, train_image in enumerate(train_image_batch):  # run from here (included)
        image = train_image["image"].numpy().transpose(1, 2, 0)
        original = cv2.imread(train_image["file_name"])

        # original
        original_visualizer = Visualizer(
            original[:, :, ::-1], metadata=metadata, scale=1,
            instance_mode=ColorMode.IMAGE
        )
        original_img = original_visualizer.draw_dataset_dict(train_dict[train_image["file_name"]])

        # visualize ground truth
        aug_visualizer = Visualizer(
            image[:, :, ::-1], metadata=metadata, scale=1,
            instance_mode=ColorMode.IMAGE
        )
        aug_image = aug_visualizer.overlay_instances(
            boxes=train_image["instances"].gt_boxes,
            labels=train_image["instances"].gt_classes,
            masks=train_image["instances"].gt_masks,
        )

        cv2.imshow('augmented', aug_image.get_image()[:, :, ::-1])
        cv2.imshow('original', original_img.get_image()[:, :, ::-1])
        k = cv2.waitKey(0)

        # exit loop if esc is pressed
        if k == 27:
            cv2.destroyAllWindows()
            break
