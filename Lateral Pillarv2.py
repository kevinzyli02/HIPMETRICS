import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from femoral_head_chat import restore_femoral_head_curved
from lateral_pillar_utils_v2 import compute_lateral_pillar

# === Paths ===
annotation_file = 'C:/Users/SR207348/PyCharmMiscProject/.venv/HIPMETRICS/Annotations/output ipsg106.json'
image_dir = r'C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg106\ipsg106'

# === Load COCO Annotations ===
coco = COCO(annotation_file)
image_ids = coco.getImgIds()
print(f"Number of images: {len(image_ids)}")
plt.close('all')

# === Process First 10 Images ===
for i in range(min(3, len(image_ids))):
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

    # === Find femoral head mask (you may need to filter by category if multiple) ===
    found_valid_mask = False
    for ann in annotations:
        segmentation = ann['segmentation']
        height, width = img_info['height'], img_info['width']
        rle = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(rle).squeeze()

        if np.sum(mask) < 100:  # Skip if mask is too small
            continue

        found_valid_mask = True

        # === Flip mask horizontally if right hip ===
        filename = img_info['file_name'].lower()
        is_right_hip = 'r' in filename and not 'l' in filename  # Basic check: contains 'r' but not 'l'
        if is_right_hip:
            mask = np.fliplr(mask)
        plt.subplot(1, 3, 1)
        plt.imshow(mask)
        # === Femoral Head Reconstruction ===
        restored_mask = restore_femoral_head_curved(mask, visualize_steps=False)
        plt.subplot(1, 3, 2)
        plt.imshow(restored_mask)
        # Flip restored mask back if it was flipped before
        if is_right_hip:
            restored_mask = np.fliplr(restored_mask)
        plt.subplot(1,3,3)
        plt.imshow(restored_mask)
        plt.show()
        # === Compute and Visualize Lateral Pillar ===
        print(f"\n--- Processing Image {i+1}: {img_info['file_name']} ---")
        result = compute_lateral_pillar(mask, img_info['file_name'], visualize=True, restored_mask=restored_mask)
        print(result)
        break  # process only the first valid segmentation per image

    if not found_valid_mask:
        print(f"[Info] No valid femoral head mask found for image: {img_info['file_name']}")
