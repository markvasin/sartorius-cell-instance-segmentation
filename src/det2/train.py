import os

import wandb
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from src.Trainer import MyTrainer
from src.det2.swin.config import add_swins_config

setup_logger()

cfg = get_cfg()

add_swins_config(cfg)

cfg.merge_from_file("swin/config/mask_rcnn_swinS_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATASETS.TEST = ("sartorius_val",)
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.TEST.EVAL_PERIOD = 250
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "swin/config/mask_rcnn_swint_S_coco17.pth"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.003  # pick a good LR
cfg.SOLVER.MAX_ITER = 7500
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.STEPS = [5000]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

wandb.init(project='sartorius', entity='mvsmark', config=cfg, sync_tensorboard=True)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
