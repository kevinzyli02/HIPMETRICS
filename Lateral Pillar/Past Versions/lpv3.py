import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from femoral_head_chat import restore_femoral_head_curved
from lateral_pillar_utils_v2 import compute_lateral_pillar
from femoral_head_reconstruction_spherical import restore_femoral_head_spherical

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
for i in range(min(10, len(image_ids))):
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
        segmentation = ann['segmentation']
        height, width = img_info['height'], img_info['width']
        rle = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(rle).squeeze()

        if np.sum(mask) < 100:  # Skip if mask is too small
            continue

        found_valid_mask = True

        # === Flip mask horizontally if right hip ===
        filename = img_info['file_name'].lower()
        is_right_hip = 'r' in filename and not 'l' in filename  # Basic check

        # Preserve original mask for final measurement
        original_mask = mask.copy()

        # Flip temporarily for processing if right hip
        if is_right_hip:
            mask = np.fliplr(mask)

        # === Femoral Head Reconstruction ===
        restored_mask = restore_femoral_head_curved(mask, visualize_steps=False)

        # Flip restored mask back if flipped before
        if is_right_hip:
            restored_mask = np.fliplr(restored_mask)
            mask = np.fliplr(mask)  # Also flip actual mask back

        # === Visualization for debugging ===
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 3, 1)
        # plt.imshow(original_mask, cmap='gray')
        # plt.title("Original Mask")
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(mask, cmap='gray')
        # plt.title("Actual Mask (Aligned)")
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(restored_mask, cmap='Reds')
        # plt.title("Restored Mask")
        # plt.axis('off')
        #        # plt.tight_layout()
        # plt.show()

        # === Compute and Visualize Lateral Pillar ===
        print(f"\n--- Processing Image {i+1}: {img_info['file_name']} ---")
        result = compute_lateral_pillar(
            mask,
            img_info['file_name'],
            image=image_rgb,
            visualize=True,
            restored_mask=restored_mask,
            output_dir = output_dir,
        )
        print(result)
        break  # process only the first valid segmentation per image

    if not found_valid_mask:
        print(f"[Info] No valid femoral head mask found for image: {img_info['file_name']}")
