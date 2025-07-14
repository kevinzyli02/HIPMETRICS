import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from sklearn.decomposition import PCA

# === Setup Paths ===
coco_path = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
image_dir = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
output_dir = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\overlayed"
os.makedirs(output_dir, exist_ok=True)

def extract_metadata(filename):
    parts = os.path.basename(filename).split('.')[0].split('_')
    return {
        'patient': parts[1],
        'view': parts[2],
        'time': parts[3],
        'laterality': parts[5]
    }

def load_mask(coco, image_id, category_name="head"):
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        if coco.cats[ann['category_id']]['name'] == category_name:
            seg = ann['segmentation']
            height = coco.imgs[image_id]['height']
            width = coco.imgs[image_id]['width']
            if isinstance(seg, list):
                rle = maskUtils.frPyObjects(seg, height, width)
                rle = maskUtils.merge(rle)
            else:
                rle = seg
            return maskUtils.decode(rle)
    return None

def rotate_mask_horizontal(mask):
    coords = np.column_stack(np.where(mask > 0))
    pca = PCA(n_components=2)
    pca.fit(coords)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle_deg = np.rad2deg(angle)
    center = tuple(np.array(mask.shape[::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(mask.astype(np.uint8), M, (mask.shape[1], mask.shape[0]))
    return rotated

def get_height_width(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return 0, 0
    return ys.max() - ys.min(), xs.max() - xs.min()

def pad_to_center(mask, min_size=(512, 512)):
    h, w = mask.shape
    H = max(h, min_size[0])
    W = max(w, min_size[1])
    canvas = np.zeros((H, W), dtype=np.uint8)
    y_off = (H - h) // 2
    x_off = (W - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = mask
    return canvas

def overlay_and_save(mask1, mask2, label1, label2, filename):
    h, w = mask1.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask1 > 0] = [255, 0, 0]  # Red: Affected
    canvas[mask2 > 0] = [0, 255, 0]  # Green: Healthy
    plt.imshow(canvas)
    plt.title(f"{label1} (Red) vs {label2} (Green)")
    plt.axis('off')
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def process_images(coco_path, image_dir, max_patients=10):
    coco = COCO(coco_path)
    images = coco.loadImgs(coco.getImgIds())
    grouped = {}

    for img in images:
        meta = extract_metadata(img['file_name'])
        key = (meta['patient'], meta['view'], meta['time'])
        grouped.setdefault(key, {})[meta['laterality']] = img

    for i, ((patient, view, time), sides) in enumerate(grouped.items()):
        if 'L' not in sides or 'R' not in sides:
            continue
        if i >= max_patients:
            break

        img_L, img_R = sides['L'], sides['R']
        mask_L = load_mask(coco, img_L['id'])
        mask_R = load_mask(coco, img_R['id'])
        if mask_L is None or mask_R is None:
            continue

        mask_L = rotate_mask_horizontal(mask_L)
        mask_R = rotate_mask_horizontal(mask_R)

        hL, wL = get_height_width(mask_L)
        hR, wR = get_height_width(mask_R)

        ratio_L = hL / wL if wL > 0 else 0
        ratio_R = hR / wR if wR > 0 else 0

        H = max(mask_L.shape[0], mask_R.shape[0], 512)
        W = max(mask_L.shape[1], mask_R.shape[1], 512)

        if ratio_L < ratio_R:
            affected = np.fliplr(pad_to_center(mask_L, (H, W)))
            healthy = pad_to_center(mask_R, (H, W))
            labels = ('Affected (L)', 'Healthy (R)')
        else:
            affected = np.fliplr(pad_to_center(mask_R, (H, W)))
            healthy = pad_to_center(mask_L, (H, W))
            labels = ('Affected (R)', 'Healthy (L)')

        fname = f"overlay_{patient}_{view}_{time}.png"
        overlay_and_save(affected, healthy, labels[0], labels[1], fname)
        print(f"Saved overlay: {fname}")

# === Run ===
process_images(coco_path, image_dir, max_patients=10)