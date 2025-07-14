import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pathlib import Path

# Define folder locations
coco_path = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
image_dir = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
output_dir = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\overlayed"


def rotate_mask(mask, angle, center):
    """Rotate mask around specified center point"""
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), rotation_mat,
                                  (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_NEAREST)
    return rotated_mask > 0.5


def align_epiphysis(mask):
    """Rotate mask so epiphysis is oriented downward"""
    # Find centroid
    points = np.argwhere(mask)
    if len(points) == 0:
        return mask, (0, 0)
    cy, cx = np.mean(points, axis=0)

    # Find farthest point from centroid (epiphysis)
    vectors = points - np.array([cy, cx])
    distances = np.linalg.norm(vectors, axis=1)
    farthest_idx = np.argmax(distances)
    fy, fx = points[farthest_idx]

    # Calculate rotation angle (align vector to vertical downward)
    dx = fx - cx
    dy = fy - cy
    target_angle = np.degrees(np.arctan2(dx, dy))

    # Rotate mask
    rotated = rotate_mask(mask, target_angle, (cx, cy))
    return rotated, (cx, cy)


def process_patient(patient_id, timepoint, view, coco, image_dir):
    """Process left/right masks for a specific patient/timepoint/view"""
    # Find matching annotations
    anns = []
    for img_id in coco.imgs:
        img_info = coco.imgs[img_id]
        filename = Path(img_info['file_name']).stem
        parts = filename.split('_')

        if (parts[0] == f"Patient{patient_id}" and
                parts[2] == timepoint and
                parts[1] == view and
                parts[-1] in ['L', 'R']):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            for ann_id in ann_ids:
                ann = coco.anns[ann_id]
                if coco.cats[ann['category_id']]['name'] == 'femoral head':
                    anns.append({
                        'mask': coco.annToMask(ann),
                        'laterality': parts[-1],
                        'img_size': (img_info['width'], img_info['height'])
                    })

    # Process masks
    results = {}
    for ann in anns:
        mask = ann['mask']
        laterality = ann['laterality']

        # Mirror left masks to match right orientation
        if laterality == 'L':
            mask = cv2.flip(mask, 1)

        # Align epiphysis to bottom
        aligned_mask, centroid = align_epiphysis(mask)
        results[laterality] = {'mask': aligned_mask, 'centroid': centroid}

    return results


def overlay_masks(left_data, right_data):
    """Overlay left and right masks with centroids aligned"""
    # Create blank canvas
    canvas_size = 1500
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    center = (canvas_size // 2, canvas_size // 2)

    # Place masks
    for i, data in enumerate([right_data, left_data]):
        mask = data['mask']
        cx, cy = data['centroid']

        # Calculate position
        x_start = center[0] - cx
        y_start = center[1] - cy
        x_end = x_start + mask.shape[1]
        y_end = y_start + mask.shape[0]

        # Clip coordinates
        x1 = max(0, x_start)
        y1 = max(0, y_start)
        x2 = min(canvas_size, x_end)
        y2 = min(canvas_size, y_end)

        if x1 >= x2 or y1 >= y2:
            continue

        # Extract mask region
        mx1 = x1 - x_start
        my1 = y1 - y_start
        mx2 = mx1 + (x2 - x1)
        my2 = my1 + (y2 - y1)
        mask_region = mask[my1:my2, mx1:mx2]

        # Set color (right=red, left=green)
        color = [0, 255, 0] if i == 1 else [255, 0, 0]
        canvas[y1:y2, x1:x2, :] = np.where(
            mask_region[..., None],
            np.array(color, dtype=np.uint8),
            canvas[y1:y2, x1:x2, :]
        )

    return canvas


def save_overlay_image(overlay, patient_id, timepoint, view, output_dir):
    """Save the overlay image to the specified directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    filename = f"Patient_{patient_id}_{view}_{timepoint}_overlay.png"
    output_path = os.path.join(output_dir, filename)

    # Convert BGR to RGB for saving with matplotlib
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Save image
    cv2.imwrite(output_path, overlay_rgb)
    print(f"Saved overlay image to: {output_path}")


def main():
    # Initialize COCO API
    coco = COCO(coco_path)

    # Find all unique patient/timepoint combinations
    patients = set()
    for img_id in coco.imgs:
        filename = Path(coco.imgs[img_id]['file_name']).stem
        parts = filename.split('_')
        patients.add((parts[0][7:], parts[2], parts[1]))  # (patient_id, timepoint, view)

    # Process each patient/timepoint
    for patient_id, timepoint, view in patients:
        results = process_patient(patient_id, timepoint, view, coco, image_dir)

        if 'L' not in results or 'R' not in results:
            print(f"Skipping Patient {patient_id} {timepoint} {view} - missing left or right mask")
            continue

        # Create overlay visualization
        overlay = overlay_masks(results['L'], results['R'])

        # Save overlay image
        save_overlay_image(overlay, patient_id, timepoint, view, output_dir)

        # Display results
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Patient {patient_id} | {timepoint} | {view}\nRed: Right, Green: Left (Mirrored)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()