from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

TRAIN_NAME = "custom_train"
VAL_NAME = "custom_val"

# Register datasets
if TRAIN_NAME not in DatasetCatalog.list():
    register_coco_instances(
        TRAIN_NAME, {},
        "potato_coco_seg/train/_annotations.coco.json",
        "potato_coco_seg/train/images"
    )

if VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(
        VAL_NAME, {},
        "potato_coco_seg/val/_annotations.coco.json",
        "potato_coco_seg/val/images"
    )

# Define class names
class_names = [
    "Potato_growth_crack", "Potato_tuber_defect", "Potato_tuber_greening",
    "Potato_tuber_knobby", "Potato_tuber_regular", "Sprout"
]

MetadataCatalog.get(TRAIN_NAME).thing_classes = class_names
MetadataCatalog.get(VAL_NAME).thing_classes = class_names

print("? Datasets registered successfully!")

