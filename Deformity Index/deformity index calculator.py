import os
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.transform import resize
from skimage.measure import regionprops, label

def load_coco_masks(coco_json_path, image_folder):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = defaultdict(list)

    for ann in coco_data['annotations']:
        if 'head' in coco_data['categories'][ann['category_id'] - 1]['name'].lower():
            annotations[ann['image_id']].append(ann)

    image_data = {}
    for image_id, anns in annotations.items():
        img_info = images[image_id]
        img_path = os.path.join(image_folder, img_info['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        masks = []
        for ann in anns:
            rle = maskUtils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
            mask = maskUtils.decode(rle)
            if mask.ndim > 2:
                mask = mask[..., 0]
            masks.append(mask)

        if len(masks) == 1:
            image_data[img_info['file_name']] = {'image': img, 'mask': masks[0], 'info': img_info}

    return image_data

def is_affected(mask):
    props = regionprops(label(mask))
    if not props:
        return False
    height = props[0].bbox[2] - props[0].bbox[0]
    width = props[0].bbox[3] - props[0].bbox[1]
    return height < width

def bottom_left(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] == 0:
        return np.array([0.0, 0.0])
    y_max = np.max(coords[:, 0])
    x_min = np.min(coords[coords[:, 0] == y_max, 1])
    return np.array([x_min, y_max], dtype=np.float32)

def rotate_mask_around_point(mask, angle, point):
    point = tuple(map(float, point))
    M = cv2.getRotationMatrix2D(point, angle, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, (mask.shape[1], mask.shape[0]))

def rigid_align_bottom_left(left_mask, right_mask):
    affected_is_left = is_affected(left_mask)
    if affected_is_left:
        right_mask = np.fliplr(right_mask)
    else:
        left_mask = np.fliplr(left_mask)

    bl_left = bottom_left(left_mask)
    bl_right = bottom_left(right_mask)
    delta = bl_left - bl_right
    angle = 0.0 if np.linalg.norm(delta) == 0 else np.degrees(np.arctan2(delta[1], delta[0]))

    left_rotated = rotate_mask_around_point(left_mask, angle, bl_left)
    right_rotated = rotate_mask_around_point(right_mask, angle, bl_left)

    return left_rotated, right_rotated, bl_left, angle

def compute_deformity_index(left_mask, right_mask):
    h = min(left_mask.shape[0], right_mask.shape[0])
    w = min(left_mask.shape[1], right_mask.shape[1])
    left_mask = left_mask[:h, :w]
    right_mask = right_mask[:h, :w]

    x_diffs, y_diffs = [], []
    for y in range(h):
        lx = np.where(left_mask[y, :] > 0)[0]
        rx = np.where(right_mask[y, :] > 0)[0]
        if lx.size > 0 and rx.size > 0:
            x_diffs.append(np.abs(lx[-1] - rx[-1]))
    for x in range(w):
        ly = np.where(left_mask[:, x] > 0)[0]
        ry = np.where(right_mask[:, x] > 0)[0]
        if ly.size > 0 and ry.size > 0:
            y_diffs.append(np.abs(ly[-1] - ry[-1]))

    max_x_diff = max(x_diffs) if x_diffs else 0
    max_y_diff = max(y_diffs) if y_diffs else 0

    def get_diameter(mask):
        coords = np.column_stack(np.where(mask > 0))
        x_diam = np.ptp(coords[:, 1]) if coords.shape[0] else 0
        y_diam = np.ptp(coords[:, 0]) if coords.shape[0] else 0
        return max(x_diam, y_diam)

    d1 = get_diameter(left_mask)
    d2 = get_diameter(right_mask)
    d = min(d1, d2)
    return (max_x_diff + max_y_diff) / d if d > 0 else None

def align_and_calculate_DI(left_mask, right_mask, visualize=False, output_path=None, title=""):
    aligned_left, aligned_right, ref_point, angle = rigid_align_bottom_left(left_mask, right_mask)
    DI = compute_deformity_index(aligned_left, aligned_right)

    if visualize and output_path:
        h = min(aligned_left.shape[0], aligned_right.shape[0])
        w = min(aligned_left.shape[1], aligned_right.shape[1])
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[aligned_left[:h, :w] > 0] = [255, 0, 0]
        overlay[aligned_right[:h, :w] > 0] = [0, 255, 0]
        overlay[(aligned_left[:h, :w] > 0) & (aligned_right[:h, :w] > 0)] = [255, 255, 0]

        plt.imshow(overlay)
        plt.plot(ref_point[0], ref_point[1], 'bo')
        plt.title(f"{title}\nRigid Align (Bottom-Left)\nDI: {DI:.3f}, Angle: {angle:.2f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_path, f"{title}_rigid_align_overlay.png"))
        plt.close()

    return DI

def run_deformity_index_pipeline(coco_json_path, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_data = load_coco_masks(coco_json_path, image_folder)
    grouped = defaultdict(dict)

    for fname in image_data:
        key = "_".join(fname.split('_')[:-1])
        laterality = fname.split('_')[-1].split('.')[0]
        grouped[key][laterality] = image_data[fname]['mask']

    results = []
    for i, (key, sides) in enumerate(grouped.items()):
        if i >= 10:
            break
        if 'L' in sides and 'R' in sides:
            di = align_and_calculate_DI(
                left_mask=sides['L'], right_mask=sides['R'],
                visualize=True, output_path=output_folder, title=key
            )
            results.append((key, di))

    return results

results = run_deformity_index_pipeline(
    coco_json_path=r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json',
    image_folder=r'C:\Users\SR207348\Downloads\ipsg102\ipsg102',
    output_folder=r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_DI"
)

for key, di in results:
    print(f"{key}: DI = {di:.3f}")
