import os

from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from src.augmentations import get_dataset_mapper
from src.evaluator import CustomEvaluator, LossEvalHook

setup_logger()


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CustomEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=get_dataset_mapper(cfg))
