import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# Dataset names
TRAIN_NAME = "custom_train"
VAL_NAME = "custom_val"

# Only register if not already done
if TRAIN_NAME not in DatasetCatalog.list():
    register_coco_instances(TRAIN_NAME, {}, 
        "train/_annotations.coco.json", 
        "train")

if VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(VAL_NAME, {}, 
        "valid/_annotations.coco.json", 
        "valid")

# Config setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (TRAIN_NAME,)
cfg.DATASETS.TEST = (VAL_NAME,)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (12000, 13500)  # No LR decay

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # YOUR number of object classes

cfg.OUTPUT_DIR = "./output2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Trainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluation after training
evaluator = COCOEvaluator(VAL_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, VAL_NAME)
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print("?? Evaluation Metrics:")
print(metrics)

