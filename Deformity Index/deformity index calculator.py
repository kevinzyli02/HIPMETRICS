import os
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from collections import defaultdict
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

def get_centroid(mask):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return np.array([0, 0])
    return np.array([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])

def get_rightmost_lowest_point(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return np.array([0, 0])
    sorted_coords = coords[np.lexsort((-coords[:, 1], coords[:, 0]))]  # rightmost, then lowest
    return sorted_coords[0][::-1]  # (x, y)

def get_orientation_angle(mask):
    props = regionprops(label(mask))
    if not props:
        return 0
    # Convert orientation to degrees and invert sign to rotate correctly
    return -props[0].orientation * 180 / np.pi

def rotate_around_point(mask, angle, point):
    # Convert point to Python floats for OpenCV compatibility
    center = (float(point[0]), float(point[1]))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, (mask.shape[1], mask.shape[0]))

def translate_mask(mask, shift):
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(mask.astype(np.uint8), M, (mask.shape[1], mask.shape[0]))

def dice_coefficient(mask1, mask2):
    # Crop to minimal shape
    h = min(mask1.shape[0], mask2.shape[0])
    w = min(mask1.shape[1], mask2.shape[1])

    mask1_crop = mask1[:h, :w].astype(bool)
    mask2_crop = mask2[:h, :w].astype(bool)

    intersection = np.logical_and(mask1_crop, mask2_crop).sum()
    size_sum = mask1_crop.sum() + mask2_crop.sum()
    if size_sum == 0:
        return 1.0  # Both empty masks considered perfect overlap
    return 2.0 * intersection / size_sum

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

def align_centroid(left_mask, right_mask):
    left_c = get_centroid(left_mask)
    right_c = get_centroid(right_mask)
    shift = right_c - left_c
    aligned_left = translate_mask(left_mask, shift)
    return aligned_left, right_c, shift

def align_rightmost_lowest_no_rotation(left_mask, right_mask):
    left_p = get_rightmost_lowest_point(left_mask)
    right_p = get_rightmost_lowest_point(right_mask)
    shift = right_p - left_p
    aligned_left = translate_mask(left_mask, shift)
    return aligned_left, right_p, shift

def align_rightmost_lowest_with_rotation(left_mask, right_mask):
    left_p = get_rightmost_lowest_point(left_mask)
    angle = get_orientation_angle(left_mask)
    rotated_left = rotate_around_point(left_mask, angle, left_p)
    rotated_right = rotate_around_point(right_mask, angle, left_p)
    return rotated_left, rotated_right, left_p, angle

def visualize_alignments(left_mask, right_mask, aligned_masks, ref_points, angles, shifts, title, output_path):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    strategy_names = ["Centroid Align", "Rightmost-Lowest Align\n(No Rotation)", "Rightmost-Lowest Align\nWith Rotation"]

    for i, ax in enumerate(axs):
        lm = aligned_masks[i]
        # Right mask for first two strategies is original, for third it's rotated
        if i == 2:
            rm = aligned_masks[1]  # The rightmost no rotation aligned right mask (rotated mask aligns both)
        else:
            rm = right_mask
        h = min(lm.shape[0], rm.shape[0])
        w = min(lm.shape[1], rm.shape[1])
        lm_crop = lm[:h, :w]
        rm_crop = rm[:h, :w]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[lm_crop > 0] = [255, 0, 0]  # left in red
        overlay[rm_crop > 0] = [0, 255, 0]  # right in green
        overlay[(lm_crop > 0) & (rm_crop > 0)] = [255, 255, 0]  # overlap yellow

        # Plot alignment reference point if applicable
        rp = ref_points[i]
        if rp[0] < w and rp[1] < h:
            ax.plot(rp[0], rp[1], 'bo', markersize=8)

        ax.imshow(overlay)
        ax.set_title(strategy_names[i])
        ax.axis('off')

    plt.suptitle(f"{title}\nBlue dot = alignment reference point")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_path, f"{title}_alignment_comparison.png"))
    plt.close()

def run_deformity_index_comparison_pipeline(coco_json_path, image_folder, output_folder):
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
            left_mask = np.fliplr(sides['L'])  # Flip left side horizontally if your data requires it
            right_mask = sides['R']

            # 1. Centroid alignment
            aligned_left_c, ref_c, shift_c = align_centroid(left_mask, right_mask)
            di_c = compute_deformity_index(aligned_left_c, right_mask)
            dice_c = dice_coefficient(aligned_left_c, right_mask)

            # 2. Rightmost-lowest alignment no rotation
            aligned_left_rn, ref_rn, shift_rn = align_rightmost_lowest_no_rotation(left_mask, right_mask)
            di_rn = compute_deformity_index(aligned_left_rn, right_mask)
            dice_rn = dice_coefficient(aligned_left_rn, right_mask)

            # 3. Rightmost-lowest alignment with rotation
            rotated_left_rr, rotated_right_rr, ref_rr, angle_rr = align_rightmost_lowest_with_rotation(left_mask, right_mask)
            di_rr = compute_deformity_index(rotated_left_rr, rotated_right_rr)
            dice_rr = dice_coefficient(rotated_left_rr, rotated_right_rr)

            aligned_masks = [aligned_left_c, aligned_left_rn, rotated_left_rr]
            ref_points = [ref_c, ref_rn, ref_rr]
            angles = [0, 0, angle_rr]
            shifts = [shift_c, shift_rn, None]

            visualize_alignments(left_mask, right_mask, aligned_masks, ref_points, angles, shifts, key, output_folder)
            plt.show
            results.append({
                'key': key,
                'DI_centroid': di_c,
                'Dice_centroid': dice_c,
                'DI_rightmost_no_rot': di_rn,
                'Dice_rightmost_no_rot': dice_rn,
                'DI_rightmost_rot': di_rr,
                'Dice_rightmost_rot': dice_rr,
                'rotation_angle': angle_rr,
                'ref_point': ref_rr.tolist()
            })

    return results

# Run the comparison pipeline:

results = run_deformity_index_comparison_pipeline(
    coco_json_path=r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json',
    image_folder=r'C:\Users\SR207348\Downloads\ipsg102\ipsg102',
    output_folder=r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_DI_comparison"
)

# Print summary:

for res in results:
    print(f"{res['key']}:")
    print(f"  Centroid Align     -> DI: {res['DI_centroid']:.3f}, Dice: {res['Dice_centroid']:.3f}")
    print(f"  Rightmost No Rot   -> DI: {res['DI_rightmost_no_rot']:.3f}, Dice: {res['Dice_rightmost_no_rot']:.3f}")
    print(f"  Rightmost With Rot -> DI: {res['DI_rightmost_rot']:.3f}, Dice: {res['Dice_rightmost_rot']:.3f}")
    print(f"  Rotation Angle: {res['rotation_angle']:.2f} deg, Ref Point: {res['ref_point']}")
    print()
