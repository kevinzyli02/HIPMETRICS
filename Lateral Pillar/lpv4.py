import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from femoral_head_chat import restore_femoral_head_curved
from lateral_pillar_utils_v2 import compute_lateral_pillar

# === Paths ===
annotation_file = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
image_dir = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
output_dir = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_lateral_pillar"

# === Load COCO Annotations ===
coco = COCO(annotation_file)
image_ids = coco.getImgIds()
print(f"Number of images: {len(image_ids)}")
plt.close('all')

# === Process First 10 Images ===
for i in range(min(100, len(image_ids))):
    img_info = coco.loadImgs(image_ids[i])[0]
    image_path = os.path.join(image_dir, img_info['file_name'])

    # === Read Image ===
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Warning] Could not read image at: {image_path}")
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === Get Femoral Head Annotation ===
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    annotations = coco.loadAnns(ann_ids)
    if len(annotations) == 0:
        print(f"[Info] No annotations found for image: {img_info['file_name']}")
        continue

    found_valid_mask = False
    for ann in annotations:
        # Ensure category name or ID matches 'head'
        if 'category_id' in ann:
            category_info = coco.loadCats([ann['category_id']])[0]
            if 'head' not in category_info['name'].lower():
                continue

        segmentation = ann['segmentation']
        height, width = img_info['height'], img_info['width']
        rle = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(rle).squeeze()

        if np.sum(mask) < 100:
            continue

        found_valid_mask = True

        filename = img_info['file_name'].lower()
        is_right_hip = 'r' in filename and 'l' not in filename

        original_mask = mask.copy()
        if is_right_hip:
            mask = np.fliplr(mask)

        restored_mask = restore_femoral_head_curved(mask, visualize_steps=False)

        if is_right_hip:
            restored_mask = np.fliplr(restored_mask)
            mask = np.fliplr(mask)

        print(f"\n--- Processing Image {i+1}: {img_info['file_name']} ---")
        result = compute_lateral_pillar(
            mask,
            img_info['file_name'],
            image=image_rgb,
            visualize=True,
            restored_mask=restored_mask,
            output_dir=output_dir,
        )

        break  # Only process the first valid head mask

    if not found_valid_mask:
        print(f"[Info] No valid femoral head mask found for image: {img_info['file_name']}")
