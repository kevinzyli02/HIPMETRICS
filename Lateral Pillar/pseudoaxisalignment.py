import json
import numpy as np
from skimage.draw import polygon
from skimage.measure import find_contours, regionprops
from scipy.spatial import ConvexHull, distance_matrix
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib.lines import Line2D

# Use non-interactive backend for server-side execution
matplotlib.use('Agg')


def poly_to_mask(poly, width, height):
    poly = np.array(poly).reshape(-1, 2)
    if len(poly) == 0:
        return np.zeros((height, width), dtype=bool)
    x = poly[:, 0]
    y = poly[:, 1]
    rr, cc = polygon(y, x, (height, width))
    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    return mask


def get_major_minor_axis(mask):
    points_rc = np.argwhere(mask)
    if len(points_rc) < 2:
        return None, None, None, None
    points_xy = points_rc[:, [1, 0]]

    try:
        hull = ConvexHull(points_xy)
    except:
        return None, None, None, None
    hull_points = points_xy[hull.vertices]

    if len(hull_points) < 2:
        return None, None, None, None
    dist_mat = distance_matrix(hull_points, hull_points)
    i, j = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
    p1_xy = hull_points[i]
    p2_xy = hull_points[j]

    major_vector = p2_xy - p1_xy
    center_xy = (p1_xy + p2_xy) / 2.0
    major_length = np.linalg.norm(major_vector)

    if major_length == 0:
        return center_xy, major_vector, 0, 0

    u_major = major_vector / major_length
    u_minor = np.array([-u_major[1], u_major[0]])

    vectors = points_xy - center_xy
    proj = np.dot(vectors, u_minor)
    minor_length = np.max(proj) - np.min(proj)

    return center_xy, u_major, minor_length, major_length


def get_center_of_mass(mask):
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return None
    com_y, com_x = props[0].centroid
    return np.array([com_x, com_y])


def com_side_of_axis(com, center, axis_vector):
    """Determine which side of the axis the COM is on"""
    com_vector = com - center
    projection = np.dot(com_vector, axis_vector)
    return 1 if projection >= 0 else -1


def align_femoral_heads(coco_json_path, image_folder, output_folder):
    with open(coco_json_path) as f:
        coco = json.load(f)

    cat_name_to_id = {cat['name']: cat['id'] for cat in coco['categories']}
    if 'head' not in cat_name_to_id:
        raise ValueError("Category 'head' not found in COCO categories")
    head_cat_id = cat_name_to_id['head']

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Group images by patient and timepoint
    patient_timepoint_map = {}
    for img in coco['images']:
        parts = img['file_name'].split('_')
        if len(parts) >= 5 and parts[-1].split('.')[0] in ['R', 'L']:
            patient_id = parts[1]
            timepoint = '_'.join(parts[3:5])
            key = (patient_id, timepoint)
            if key not in patient_timepoint_map:
                patient_timepoint_map[key] = []
            patient_timepoint_map[key].append(img)

    # Process each patient/timepoint pair
    for (patient_id, timepoint), images in tqdm(patient_timepoint_map.items(), desc="Processing patients"):
        if len(images) != 2:
            continue

        # Process both femoral heads
        heads = []
        for img_info in images:
            anns = [ann for ann in coco['annotations']
                    if ann['image_id'] == img_info['id'] and ann['category_id'] == head_cat_id]

            if not anns:
                continue

            ann = anns[0]  # Assume one head annotation per image
            width = img_info['width']
            height = img_info['height']
            poly = ann['segmentation'][0]
            mask = poly_to_mask(poly, width, height)

            center_xy, u_major, minor_length, major_length = get_major_minor_axis(mask)
            if center_xy is None:
                continue

            # Get contour for visualization
            contours = find_contours(mask, 0.5)
            if not contours:
                continue
            contour = max(contours, key=lambda c: len(c))[:, [1, 0]]  # Convert to (x,y)

            # Get center of mass
            com = get_center_of_mass(mask)
            if com is None:
                continue

            heads.append({
                'image_info': img_info,
                'mask': mask,
                'center_xy': center_xy,
                'u_major': u_major,
                'ratio': minor_length / major_length if major_length > 0 else 0,
                'laterality': img_info['file_name'].split('_')[-1].split('.')[0],
                'contour': contour,
                'com': com  # Center of Mass (x, y)
            })

        if len(heads) != 2:
            continue

        # Identify affected and unaffected heads
        if heads[0]['ratio'] > heads[1]['ratio']:
            unaffected = heads[0]
            affected = heads[1]
        else:
            unaffected = heads[1]
            affected = heads[0]

        # Calculate rotation angle for major axis alignment
        angle_aff = np.arctan2(affected['u_major'][1], affected['u_major'][0])
        angle_unaff = np.arctan2(unaffected['u_major'][1], unaffected['u_major'][0])
        theta = angle_aff - angle_unaff

        # Normalize angle to [-π, π]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Create rotation matrices for both orientations
        R0 = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        R180 = np.array([[np.cos(theta + np.pi), -np.sin(theta + np.pi)],
                         [np.sin(theta + np.pi), np.cos(theta + np.pi)]])

        # Calculate COM distances for both orientations
        com_unaff_trans0 = R0 @ (unaffected['com'] - unaffected['center_xy']) + affected['center_xy']
        com_unaff_trans180 = R180 @ (unaffected['com'] - unaffected['center_xy']) + affected['center_xy']

        dist0 = np.linalg.norm(com_unaff_trans0 - affected['com'])
        dist180 = np.linalg.norm(com_unaff_trans180 - affected['com'])

        # Choose orientation with smallest COM distance
        if dist0 <= dist180:
            R = R0
            com_unaff_trans = com_unaff_trans0
            orientation_used = "0°"
        else:
            R = R180
            com_unaff_trans = com_unaff_trans180
            orientation_used = "180°"

        # Transform unaffected head to affected head's space
        transformed_contour = []
        center_unaff_xy = unaffected['center_xy']
        center_aff_xy = affected['center_xy']
        aff_height = affected['image_info']['height']
        aff_width = affected['image_info']['width']

        for point in unaffected['contour']:
            P_xy = np.array([point[0], point[1]])
            P_xy_transformed = R @ (P_xy - center_unaff_xy) + center_aff_xy
            transformed_contour.append([P_xy_transformed[1], P_xy_transformed[0]])

        transformed_contour = np.array(transformed_contour)
        rr, cc = polygon(transformed_contour[:, 0], transformed_contour[:, 1], (aff_height, aff_width))
        transformed_unaff_mask = np.zeros((aff_height, aff_width), dtype=bool)
        transformed_unaff_mask[rr, cc] = True

        # Load original images for visualization
        affected_img_path = os.path.join(image_folder, affected['image_info']['file_name'])
        unaffected_img_path = os.path.join(image_folder, unaffected['image_info']['file_name'])

        affected_img = np.array(Image.open(affected_img_path).convert('L'))
        unaffected_img = np.array(Image.open(unaffected_img_path).convert('L'))

        # Create 2x3 comparison figure
        fig, ax = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Patient {patient_id} - {timepoint} | Orientation: {orientation_used} | Dist: {min(dist0, dist180):.1f}",
            fontsize=16)

        # Create custom legend handles
        red_line = Line2D([0], [0], color='red', linewidth=2, label='Affected')
        blue_dashed = Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Unaffected')
        com_marker = Line2D([0], [0], marker='x', color='cyan', markersize=8, label='CoM', linestyle='')
        axis_vector = Line2D([0], [0], color='lime', linewidth=3, label='Major Axis')
        trans_com_marker = Line2D([0], [0], marker='o', color='magenta', markersize=8,
                                  markerfacecolor='none', linestyle='', label='Trans COM')

        # Original heads
        ax[0, 0].imshow(affected_img, cmap='gray')
        ax[0, 0].plot(affected['contour'][:, 0], affected['contour'][:, 1], 'r-', linewidth=1)
        ax[0, 0].scatter(affected['com'][0], affected['com'][1], facecolor='cyan', edgecolor='cyan',
                         s=80, marker='x')
        ax[0, 0].set_title(f"Affected Head ({affected['laterality']})")
        ax[0, 0].legend(handles=[com_marker], loc='upper right')
        ax[0, 0].axis('off')

        ax[0, 1].imshow(unaffected_img, cmap='gray')
        ax[0, 1].plot(unaffected['contour'][:, 0], unaffected['contour'][:, 1], 'b-', linewidth=1)
        ax[0, 1].scatter(unaffected['com'][0], unaffected['com'][1], facecolor='cyan', edgecolor='cyan',
                         s=80, marker='x')
        ax[0, 1].set_title(f"Unaffected Head ({unaffected['laterality']})")
        ax[0, 1].legend(handles=[com_marker], loc='upper right')
        ax[0, 1].axis('off')

        # Aligned heads overlay
        ax[0, 2].imshow(affected_img, cmap='gray')
        ax[0, 2].contour(affected['mask'], colors='red', linewidths=2)
        ax[0, 2].contour(transformed_unaff_mask, colors='blue', linewidths=2, linestyles='dashed')
        ax[0, 2].scatter(com_unaff_trans[0], com_unaff_trans[1], facecolor='none', edgecolor='magenta',
                         s=80, marker='o')
        ax[0, 2].set_title('Aligned Overlay with Transformed COM')
        ax[0, 2].legend(handles=[red_line, blue_dashed, trans_com_marker], loc='upper right')
        ax[0, 2].axis('off')

        # Major axis visualization
        ax[1, 0].imshow(affected_img, cmap='gray')
        ax[1, 0].plot(affected['contour'][:, 0], affected['contour'][:, 1], 'r-', linewidth=1)
        ax[1, 0].scatter(affected['com'][0], affected['com'][1], facecolor='cyan', edgecolor='cyan',
                         s=80, marker='x')
        ax[1, 0].quiver(affected['center_xy'][0], affected['center_xy'][1],
                        affected['u_major'][0] * 50, affected['u_major'][1] * 50,
                        color='lime', scale=1, scale_units='xy', angles='xy', width=0.005)
        ax[1, 0].set_title('Affected: Major Axis & CoM')
        ax[1, 0].legend(handles=[axis_vector, com_marker], loc='upper right')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(unaffected_img, cmap='gray')
        ax[1, 1].plot(unaffected['contour'][:, 0], unaffected['contour'][:, 1], 'b-', linewidth=1)
        ax[1, 1].scatter(unaffected['com'][0], unaffected['com'][1], facecolor='cyan', edgecolor='cyan',
                         s=80, marker='x')
        ax[1, 1].quiver(unaffected['center_xy'][0], unaffected['center_xy'][1],
                        unaffected['u_major'][0] * 50, unaffected['u_major'][1] * 50,
                        color='lime', scale=1, scale_units='xy', angles='xy', width=0.005)
        ax[1, 1].set_title('Unaffected: Major Axis & CoM')
        ax[1, 1].legend(handles=[axis_vector, com_marker], loc='upper right')
        ax[1, 1].axis('off')

        # COM position analysis
        ax[1, 2].imshow(affected_img, cmap='gray')
        ax[1, 2].scatter(affected['com'][0], affected['com'][1], facecolor='cyan', edgecolor='cyan',
                         s=80, marker='x')
        ax[1, 2].scatter(com_unaff_trans[0], com_unaff_trans[1], facecolor='none', edgecolor='magenta',
                         s=80, marker='o')
        ax[1, 2].quiver(affected['center_xy'][0], affected['center_xy'][1],
                        affected['u_major'][0] * 50, affected['u_major'][1] * 50,
                        color='lime', scale=1, scale_units='xy', angles='xy', width=0.005)

        # Draw the major axis line
        axis_line_length = 100
        axis_start = affected['center_xy'] - affected['u_major'] * axis_line_length
        axis_end = affected['center_xy'] + affected['u_major'] * axis_line_length
        ax[1, 2].plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]],
                      color='lime', linestyle='-', linewidth=2, alpha=0.7)

        # Add distance information
        ax[1, 2].plot([affected['com'][0], com_unaff_trans[0]],
                      [affected['com'][1], com_unaff_trans[1]],
                      'w--', linewidth=1.5, alpha=0.7)
        mid_point = (affected['com'] + com_unaff_trans) / 2
        ax[1, 2].text(mid_point[0], mid_point[1], f"{min(dist0, dist180):.1f}px",
                      color='white', fontsize=10, ha='center', va='center',
                      bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

        ax[1, 2].set_title('COM Position Analysis')
        ax[1, 2].legend(handles=[com_marker, trans_com_marker, axis_vector], loc='upper right')
        ax[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save the comparison figure
        viz_path = os.path.join(output_folder, f"Patient_{patient_id}_{timepoint}_alignment.png")
        plt.savefig(viz_path, bbox_inches='tight', dpi=150)
        plt.close()

    print(f"Processing complete. Comparison figures saved to {output_folder}")


# Main execution
if __name__ == "__main__":
    coco_json_path = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
    image_folder = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
    output_folder = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_DI"

    align_femoral_heads(coco_json_path, image_folder, output_folder)