import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from lateral_pillar_utils import compute_lateral_pillar
from lateral_pillar_reference import compute_lateral_pillar_reference, overlay_hemisphere_on_mask
from femoral_head_chat import rotate_image_and_mask,get_rotation_to_horizontal,restore_femoral_head_curved
# === Paths ===
annotation_file = 'C:/Users/SR207348/PyCharmMiscProject/.venv/HIPMETRICS/Annotations/output ipsg106.json'
image_dir = r'C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg106\ipsg106'

# === Load COCO Annotations ===
coco = COCO(annotation_file)
image_ids = coco.getImgIds()
print(f"Number of images: {len(image_ids)}")

# === Load First Image Metadata ===
img_info = coco.loadImgs(image_ids[0])[0]
image_path = os.path.join(image_dir, img_info['file_name'])

# === Read Image ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Get Annotations ===
ann_ids = coco.getAnnIds(imgIds=img_info['id'])
annotations = coco.loadAnns(ann_ids)
print(f"Found {len(annotations)} segmentations")

# === Decode First Segmentation ===
height, width = img_info['height'], img_info['width']
segmentation = annotations[0]['segmentation']
rle = maskUtils.frPyObjects(segmentation, height, width)
mask = maskUtils.decode(rle).squeeze()

# === Display Image + Mask ===
#plt.figure(figsize=(8, 8), dpi=300)
#plt.imshow(image_rgb)
#plt.imshow(mask, alpha=0.5, cmap='Reds')
#plt.axis('off')
#plt.title("Femoral Head Segmentation")
#plt.show()

# === Compute Lateral Pillar ===
result = compute_lateral_pillar(mask, img_info['file_name'], visualize=False)
print(result)

# === Overlay Hemisphere on Mask ===
reference = compute_lateral_pillar_reference(mask, visualize=False)
overlay = overlay_hemisphere_on_mask(
    mask,
    radius=reference['radius'],
    lateral_third_height=reference['lateral_third_height'],
    lateral_height=result['lateral_height'],
    visualize=True
)

# === Femoral Head Reconstruction ===
restored_mask = restore_femoral_head_curved(mask, visualize_steps=True)

# # Final comparison
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap='gray')
# plt.title(f"Original (area={np.sum(mask)})")
#
# plt.subplot(1, 2, 2)
# plt.imshow(restored_mask, cmap='gray')
# plt.title(f"Restored (area={np.sum(restored_mask)})")
#
# plt.tight_layout()
# plt.show()

