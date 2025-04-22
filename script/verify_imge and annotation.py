from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import random, cv2
import matplotlib.pyplot as plt

# Register dataset
register_coco_instances(
    "custom_train", {},
    "train/_annotations.coco.json",
    "train"
)

# Load dataset and metadata
dataset_dicts = DatasetCatalog.get("custom_train")
metadata = MetadataCatalog.get("custom_train")

# Show 5 random samples
sample_count = 5
selected_samples = random.sample(dataset_dicts, min(sample_count, len(dataset_dicts)))

for i, d in enumerate(selected_samples):
    img = cv2.imread(d["file_name"])
    if img is None:
        print(f"? Image not found: {d['file_name']}")
        continue

    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    plt.figure(figsize=(10, 6))
    plt.imshow(out.get_image())
    plt.title(f"Sample {i+1}")
    plt.axis('off')
    plt.show()

#This is working fine