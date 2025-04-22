import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.visualizer import random_color
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -------------------- Configuration --------------------
VAL_NAME = "custom_val"
VAL_JSON = "valid/_annotations.coco.json"
VAL_IMAGES = "valid"
OUTPUT_DIR = "output2"
VIS_DIR = os.path.join(OUTPUT_DIR, "vis5")
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------- Dataset Registration --------------------
if VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(VAL_NAME, {}, VAL_JSON, VAL_IMAGES)

metadata = MetadataCatalog.get(VAL_NAME)
if not hasattr(metadata, "thing_classes"):
    metadata.thing_classes = [
        "Tuber-knobs-green-defect-sprout-6AP4",
        "Potato_growth_crack", "Potato_tuber_defect",
        "Potato_tuber_greening", "Potato_tuber_knobby",
        "Potato_tuber_regular", "Sprout"
    ]

# -------------------- Model Setup --------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
cfg.MODEL.DEVICE = "cuda"
cfg.DATASETS.TEST = (VAL_NAME,)

predictor = DefaultPredictor(cfg)

# -------------------- Visualization --------------------
print("\n?? Saving visual predictions...")
image_files = [f for f in os.listdir(VAL_IMAGES) if f.endswith(".jpg") or f.endswith(".png")]

# Define fixed colors per class
custom_class_colors = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Blue
    3: (255, 255, 0),    # Yellow
    4: (255, 0, 255),    # Magenta
    5: (0, 255, 255),    # Cyan
    6: (128, 128, 128),  # Gray (or a darker grey)
}

for file_name in image_files:
    img_path = os.path.join(VAL_IMAGES, file_name)
    img = cv2.imread(img_path)
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    v = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )

    vis_output = v.draw_instance_predictions(instances)
    vis_image = vis_output.get_image()[:, :, ::-1]

    # Apply consistent color per class (one color per class for both box and mask)
    for idx, (mask, cls_id) in enumerate(zip(instances.pred_masks, instances.pred_classes)):
        color = custom_class_colors.get(int(cls_id), (0, 255, 0))
        if len(color) == 4:
            color = color[:3]
        mask = mask.numpy().astype(np.uint8)
        colored_mask = np.zeros_like(img)
        colored_mask[mask == 1] = color
        vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.5, 0)

    # Concatenate original and prediction side by side
    vis_concat = np.concatenate((img, vis_image), axis=1)
    cv2.imwrite(os.path.join(VIS_DIR, file_name), vis_concat)

# -------------------- Evaluation --------------------
print("\n?? Running COCO evaluation...")
val_loader = build_detection_test_loader(cfg, VAL_NAME)
outputs_list, inputs_list = [], []

with torch.no_grad():
    for inputs in val_loader:
        outputs = predictor.model(inputs)
        outputs_list.extend(outputs)
        inputs_list.extend(inputs)

coco_results = []
for input, output in zip(inputs_list, outputs_list):
    coco_results.extend(instances_to_coco_json(output["instances"].to("cpu"), input["image_id"]))

coco_gt = COCO(VAL_JSON)
coco_dt = coco_gt.loadRes(coco_results)

coco_eval_bbox = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval_bbox.evaluate()
coco_eval_bbox.accumulate()
coco_eval_bbox.summarize()

coco_eval_segm = COCOeval(coco_gt, coco_dt, iouType="segm")
coco_eval_segm.evaluate()
coco_eval_segm.accumulate()
coco_eval_segm.summarize()

# -------------------- PR Curve Plotting --------------------
def plot_pr_curves(coco_eval: COCOeval, iou_type: str, output_name: str, class_ids: list, class_names: list):
    precisions = coco_eval.eval['precision']
    iou_threshold_index = 0  # PR @ IoU=0.5

    plt.figure()
    ap50s = []
    for idx in class_ids:
        precision = precisions[iou_threshold_index, :, idx, 0, 0]
        if precision.mean() >= 0:
            recall = np.linspace(0, 1, len(precision))
            plt.plot(recall, precision, label=f"{class_names[idx]}")
            ap50 = precision[precision > -1].mean() * 100
            ap50s.append((class_names[idx], ap50))

    # mAP@50
    map50 = np.mean([ap for _, ap in ap50s]) if ap50s else 0.0

    # Display AP@50 for each class
    ap_text = "\n".join([f"{cls}: {ap:.1f}" for cls, ap in ap50s])
    ap_text += f"\n\nmAP@50: {map50:.1f}"
    plt.gcf().text(1.01, 0.5, ap_text, fontsize=9, va='center')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({iou_type})")
    plt.legend()
    plt.grid()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{OUTPUT_DIR}/pr_curve_{output_name}.png", bbox_inches="tight")
    plt.close()

# BBox PR Curve (all classes)
all_class_names = MetadataCatalog.get(VAL_NAME).thing_classes
plot_pr_curves(coco_eval_bbox, "bbox", "bbox", list(range(len(all_class_names))), all_class_names)

# Segmentation PR Curve (only valid segm classes)
valid_cls_ids_segm = set()
for output in outputs_list:
    if output["instances"].has("pred_classes") and output["instances"].has("pred_masks"):
        valid_cls_ids_segm.update(output["instances"].pred_classes.tolist())
valid_cls_ids_segm = sorted(list(valid_cls_ids_segm))
plot_pr_curves(coco_eval_segm, "segm", "segm", valid_cls_ids_segm, all_class_names)

# -------------------- Training Diagnostics --------------------
print("\n?? Skipping mAP@50 curve plot by user request.")
