import os
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.transform import resize

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


def align_and_calculate_DI(left_mask, right_mask, visualize=False, output_path=None, title=""):
    flipped_left = np.fliplr(left_mask)

    # Resize to match unaffected mask if needed
    if flipped_left.shape != right_mask.shape:
        flipped_left = resize(flipped_left, right_mask.shape, preserve_range=True).astype(np.uint8)

    contours_left = cv2.findContours(flipped_left.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_right = cv2.findContours(right_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if len(contours_left) == 0 or len(contours_right) == 0:
        return None

    left_coords = np.vstack(contours_left).squeeze()
    right_coords = np.vstack(contours_right).squeeze()

    # Align masks using centroid
    def get_centroid(mask):
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0: return np.array([0, 0])
        return np.array([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])

    left_centroid = get_centroid(flipped_left)
    right_centroid = get_centroid(right_mask)

    shift = right_centroid - left_centroid
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    aligned_left = cv2.warpAffine(flipped_left, M, (right_mask.shape[1], right_mask.shape[0]))

    # Compute height (vertical) and width (horizontal) differences
    diff_mask = np.abs(aligned_left - right_mask)
    y_diff = np.max(np.abs(np.argmax(aligned_left, axis=0) - np.argmax(right_mask, axis=0)))
    x_diff = np.max(np.abs(np.argmax(aligned_left, axis=1) - np.argmax(right_mask, axis=1)))

    # Diameter = width of unaffected (right) mask
    right_contour = np.vstack(contours_right).squeeze()
    x_min, x_max = np.min(right_contour[:, 0]), np.max(right_contour[:, 0])
    diameter = x_max - x_min

    deformity_index = (y_diff + x_diff) / diameter if diameter > 0 else None

    if visualize and output_path:
        vis = np.zeros((*right_mask.shape, 3), dtype=np.uint8)
        vis[right_mask > 0] = [0, 255, 0]     # Right (unaffected) in green
        vis[aligned_left > 0] = [255, 0, 0]   # Left (affected) in blue
        vis[(aligned_left > 0) & (right_mask > 0)] = [255, 255, 0]

        plt.imshow(vis)
        plt.title(f"{title}\nDI: {deformity_index:.3f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_path, f"{title}_di_overlay.png"))
        plt.close()

    return deformity_index


def run_deformity_index_pipeline(coco_json_path, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_data = load_coco_masks(coco_json_path, image_folder)

    grouped = defaultdict(dict)

    # group images by patient/view/time/laterality
    for fname in image_data:
        key = "_".join(fname.split('_')[:-1])  # strip laterality (assuming it's last)
        laterality = fname.split('_')[-1].split('.')[0]  # L or R
        grouped[key][laterality] = image_data[fname]['mask']

    results = []
    for key, sides in grouped.items():
        if 'L' in sides and 'R' in sides:
            di = align_and_calculate_DI(
                left_mask=sides['L'], right_mask=sides['R'],
                visualize=True, output_path=output_folder, title=key
            )
            results.append((key, di))

    return results


results = run_deformity_index_pipeline(
    coco_json_path=r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json',
    image_folder=r'C:\Users\SR207348\Downloads\ipsg102\ipsg102' ,
    output_folder=r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_DI"
)

for key, di in results:
    print(f"{key}: DI = {di:.3f}")
#annotation_file = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
#image_dir = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
#output_dir = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_lateral_pillar"