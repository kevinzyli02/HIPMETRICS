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
output_dir = r'C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_lateral_pillar'

os.makedirs(output_dir, exist_ok=True)

# === Load COCO Annotations ===
coco = COCO(annotation_file)
image_ids = coco.getImgIds()
print(f"Number of images: {len(image_ids)}")

# === Process All Images ===
for i, img_id in enumerate(image_ids):
    img_info = coco.loadImgs(img_id)[0]
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

        if np.sum(mask) < 100:
            continue

        found_valid_mask = True

        # === Flip mask for right hip ===
        filename = img_info['file_name'].lower()
        is_right_hip = 'r' in filename and not 'l' in filename

        if is_right_hip:
            mask = np.fliplr(mask)

        restored_mask = restore_femoral_head_curved(mask, visualize_steps=False)

        if is_right_hip:
            restored_mask = np.fliplr(restored_mask)

        # === Save Visualization without plotting ===
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.imshow(image_rgb, cmap='gray')

        # Overlays
        from skimage import measure
        contours_actual = measure.find_contours(mask, 0.5)
        for contour in contours_actual:
            ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=1.5)

        contours_restored = measure.find_contours(restored_mask, 0.5)
        for contour in contours_restored:
            ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.5)

        # PCA for thirds
        ys, xs = np.where(mask > 0)
        coords = np.column_stack((xs, ys))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(coords)
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        center = np.mean(coords, axis=0)
        rot_mat = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        coords_centered = coords - center
        coords_rotated = coords_centered @ rot_mat.T
        x_rot = coords_rotated[:, 0]
        x_min, x_max = np.min(x_rot), np.max(x_rot)
        w = (x_max - x_min) / 3
        thirds = [x_min + w, x_min + 2 * w]

        for x in thirds:
            p1 = np.array([x, -100])
            p2 = np.array([x, 100])
            p1_global = (p1 @ rot_mat) + center
            p2_global = (p2 @ rot_mat) + center
            ax.plot([p1_global[0], p2_global[0]], [p1_global[1], p2_global[1]], linestyle='--', color='cyan', linewidth=1.2)

        ax.axis('off')
        ax.set_facecolor('black')
        plt.tight_layout()

        save_path = os.path.join(output_dir, img_info['file_name'].replace('.BMP', '_overlay.png'))
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        break

    if not found_valid_mask:
        print(f"[Info] No valid femoral head mask found for image: {img_info['file_name']}")
