import json
import numpy as np
from skimage.draw import polygon
from skimage.measure import regionprops
from scipy.spatial import ConvexHull, distance_matrix
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from matplotlib.lines import Line2D
import time
import cv2

# Use non-interactive backend for server-side execution
matplotlib.use('Agg')


def find_medial_lateral_points(head_mask):
    '''
    Finds the two most distant points on the femoral head contour (medial/lateral pillars).
    Parameters:
        head_mask (ndarray): Binary mask of femoral head
    Return value(s):
        point1, point2 (tuple): (x,y) coordinates of the two points
        or (None, None) if not found
    '''
    try:
        # Convert to uint8 format required by OpenCV
        mask_uint8 = (head_mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze()

        if len(contour) < 2:
            return None, None

        # Find points with maximum distance
        max_dist = -1
        pt1, pt2 = None, None
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                dist = np.linalg.norm(contour[i] - contour[j])
                if dist > max_dist:
                    max_dist = dist
                    pt1 = tuple(contour[i])
                    pt2 = tuple(contour[j])

        return pt1, pt2
    except Exception as e:
        print(f"Error finding medial/lateral points: {e}")
        return None, None


def get_major_minor_axis(mask):
    '''
    FIXED: Calculates major/minor axes using medial/lateral pillars as major axis endpoints.
            Computes minor_length and returns all required values.
    '''
    points_rc = np.argwhere(mask)
    if len(points_rc) < 2:
        return None, None, None, None, None, None

    points_xy = points_rc[:, [1, 0]]  # Convert to (x, y)

    try:
        # Use approximate convex hull for speed
        if len(points_xy) > 100:
            hull = ConvexHull(points_xy, qhull_options="QJ")
        else:
            hull = ConvexHull(points_xy)
    except:
        return None, None, None, None, None, None

    hull_points = points_xy[hull.vertices]

    if len(hull_points) < 2:
        return None, None, None, None, None, None

    # Find medial and lateral pillars (most distant points)
    dist_mat = distance_matrix(hull_points, hull_points)
    i, j = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
    p1_xy = hull_points[i]
    p2_xy = hull_points[j]

    # Major axis is between medial and lateral pillars
    major_vector = p2_xy - p1_xy
    center_xy = (p1_xy + p2_xy) / 2.0
    major_length = np.linalg.norm(major_vector)

    if major_length < 1e-5:
        return None, None, None, None, None, None

    u_major = major_vector / major_length
    u_minor = np.array([-u_major[1], u_major[0]])

    # Project hull points to minor axis
    vectors = hull_points - center_xy
    proj = np.dot(vectors, u_minor)

    # Calculate minor axis length
    minor_length = np.max(proj) - np.min(proj)

    # Find minor axis endpoints
    min_idx = np.argmin(proj)
    max_idx = np.argmax(proj)
    minor_end1 = hull_points[min_idx]
    minor_end2 = hull_points[max_idx]

    return center_xy, u_major, minor_length, major_length, (p1_xy, p2_xy), (minor_end1, minor_end2)

def poly_to_mask(poly, width, height):
    '''
    Converts polygon coordinates to a binary mask.
    '''
    if len(poly) == 0:
        return np.zeros((height, width), dtype=bool)

    poly = np.array(poly).reshape(-1, 2)
    x = poly[:, 0].clip(0, width - 1)
    y = poly[:, 1].clip(0, height - 1)
    rr, cc = polygon(y, x, (height, width))
    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    return mask


def get_center_of_mass(mask):
    '''
    Computes center of mass using region properties.
    '''
    if np.sum(mask) == 0:
        return None

    props = regionprops(mask.astype(np.uint8))
    if not props:
        return None

    com_y, com_x = props[0].centroid
    return np.array([com_x, com_y])


def find_femoral_head_pairs(coco_json_path, max_pairs=None):
    '''
    Identifies matching left/right femoral head pairs from COCO data.
    '''
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Create category mapping
    cat_name_to_id = {cat['name']: cat['id'] for cat in coco['categories']}
    if 'head' not in cat_name_to_id:
        raise ValueError("Category 'head' not found")
    head_cat_id = cat_name_to_id['head']

    # Create image ID to image mapping
    image_id_to_info = {img['id']: img for img in coco['images']}

    # Group annotations by image
    image_annotations = {}
    for ann in coco['annotations']:
        if ann['category_id'] == head_cat_id:
            img_id = ann['image_id']
            image_annotations.setdefault(img_id, []).append(ann)

    # Find femoral head pairs with limit
    pairs = []
    processed_pairs = set()

    for img_id, anns in image_annotations.items():
        if not anns:
            continue

        img_info = image_id_to_info[img_id]
        filename = img_info['file_name']

        # Extract patient ID and timepoint
        parts = filename.split('_')
        if len(parts) < 5:
            continue

        patient_id = parts[1]
        timepoint = '_'.join(parts[3:5])
        laterality = parts[-1].split('.')[0]

        if laterality not in ['L', 'R']:
            continue

        pair_key = (patient_id, timepoint)

        if pair_key in processed_pairs:
            continue

        # Find matching pair
        opposite = 'R' if laterality == 'L' else 'L'
        for other_id, other_anns in image_annotations.items():
            if other_id == img_id or not other_anns:
                continue

            other_info = image_id_to_info[other_id]
            other_file = other_info['file_name']
            other_parts = other_file.split('_')

            if len(other_parts) < 5:
                continue

            other_patient = other_parts[1]
            other_time = '_'.join(other_parts[3:5])
            other_lat = other_parts[-1].split('.')[0]

            if (other_patient == patient_id and
                    other_time == timepoint and
                    other_lat == opposite):
                pairs.append({
                    'patient_id': patient_id,
                    'timepoint': timepoint,
                    'left_img': other_info if laterality == 'L' else img_info,
                    'right_img': img_info if laterality == 'L' else other_info,
                    'left_ann': other_anns[0] if laterality == 'L' else anns[0],
                    'right_ann': anns[0] if laterality == 'L' else other_anns[0]
                })
                processed_pairs.add(pair_key)
                break

        # Apply max pairs limit
        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def process_femoral_head_pair(pair, coco_data, image_folder, visualize=False, output_folder=None):
    '''
    Processes femoral head pair by aligning unaffected head to affected head.
    '''
    # Create head dictionaries
    heads = []

    for img_key, ann_key in [('left_img', 'left_ann'), ('right_img', 'right_ann')]:
        img_info = pair[img_key]
        ann = pair[ann_key]

        width = img_info['width']
        height = img_info['height']
        poly = ann['segmentation'][0]
        mask = poly_to_mask(poly, width, height)

        # Skip if mask is empty
        if np.sum(mask) == 0:
            print(f"Empty mask found for {img_info['file_name']}")
            return None

        # Calculate major/minor axis
        axis_data = get_major_minor_axis(mask)
        if any(d is None for d in axis_data):
            print(f"Failed to calculate axis for {img_info['file_name']}")
            return None

        center_xy, u_major, minor_length, major_length, major_ends, minor_ends = axis_data

        # Get center of mass
        com = get_center_of_mass(mask)
        if com is None:
            print(f"Failed to get COM for {img_info['file_name']}")
            return None

        # Store only necessary data
        heads.append({
            'image_info': img_info,
            'mask': mask,
            'center_xy': center_xy,
            'u_major': u_major,
            'ratio': minor_length / max(major_length, 1e-5),
            'laterality': img_info['file_name'].split('_')[-1].split('.')[0],
            'com': com,
            'original_mask': mask.copy(),
            'original_center_xy': center_xy.copy(),
            'original_u_major': u_major.copy(),
            'original_com': com.copy(),
            'major_axis_ends': major_ends,
            'minor_axis_ends': minor_ends
        })

    # Identify affected and unaffected
    if heads[0]['ratio'] > heads[1]['ratio']:
        unaffected, affected = heads[0], heads[1]
    else:
        unaffected, affected = heads[1], heads[0]

    # Flip unaffected head for alignment only
    w = unaffected['image_info']['width']
    h = unaffected['image_info']['height']
    flipped_mask = np.fliplr(unaffected['original_mask'])
    flipped_center = np.array([w - unaffected['original_center_xy'][0], unaffected['original_center_xy'][1]])
    flipped_com = np.array([w - unaffected['original_com'][0], unaffected['original_com'][1]])
    flipped_u_major = np.array([-unaffected['original_u_major'][0], unaffected['original_u_major'][1]])
    flipped_major_ends = (
        np.array([w - unaffected['major_axis_ends'][0][0], unaffected['major_axis_ends'][0][1]]),
        np.array([w - unaffected['major_axis_ends'][1][0], unaffected['major_axis_ends'][1][1]])
    )
    flipped_minor_ends = (
        np.array([w - unaffected['minor_axis_ends'][0][0], unaffected['minor_axis_ends'][0][1]]),
        np.array([w - unaffected['minor_axis_ends'][1][0], unaffected['minor_axis_ends'][1][1]])
    )

    # Calculate rotation
    angle_aff = np.arctan2(affected['u_major'][1], affected['u_major'][0])
    angle_unaff = np.arctan2(flipped_u_major[1], flipped_u_major[0])
    theta = (angle_aff - angle_unaff + np.pi) % (2 * np.pi) - np.pi

    # Rotation matrices
    R0 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    R180 = np.array([[np.cos(theta + np.pi), -np.sin(theta + np.pi)],
                     [np.sin(theta + np.pi), np.cos(theta + np.pi)]])

    # COM distances
    offset = flipped_com - flipped_center
    com0 = R0 @ offset + affected['center_xy']
    com180 = R180 @ offset + affected['center_xy']

    dist0 = np.linalg.norm(com0 - affected['com'])
    dist180 = np.linalg.norm(com180 - affected['com'])

    # Choose orientation
    if dist0 <= dist180:
        R, com_trans, orientation = R0, com0, "0°"
    else:
        R, com_trans, orientation = R180, com180, "180°"

    # Transform mask using affine transformation
    aff_h, aff_w = affected['image_info']['height'], affected['image_info']['width']
    yy, xx = np.mgrid[:aff_h, :aff_w]
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    # Transform coordinates
    rel_coords = coords - affected['center_xy']
    rot_coords = (R.T @ (rel_coords).T).T + flipped_center
    trans_coords = rot_coords.reshape(aff_h, aff_w, 2)

    # Sample from flipped mask
    sample_x = trans_coords[..., 0].clip(0, w - 1)
    sample_y = trans_coords[..., 1].clip(0, h - 1)
    transformed_unaff_mask = flipped_mask[
        sample_y.astype(int),
        sample_x.astype(int)
    ]

    # Transform major axis endpoints
    trans_major_end1 = R @ (flipped_major_ends[0] - flipped_center) + affected['center_xy']
    trans_major_end2 = R @ (flipped_major_ends[1] - flipped_center) + affected['center_xy']

    # Transform minor axis endpoints
    trans_minor_end1 = R @ (flipped_minor_ends[0] - flipped_center) + affected['center_xy']
    trans_minor_end2 = R @ (flipped_minor_ends[1] - flipped_center) + affected['center_xy']

    # Calculate minor axis for transformed unaffected mask
    trans_unaff_minor_ends = (trans_minor_end1, trans_minor_end2)

    # Prepare results
    results = {
        'patient_id': pair['patient_id'],
        'timepoint': pair['timepoint'],
        'affected_mask': affected['original_mask'],
        'transformed_unaff_mask': transformed_unaff_mask,
        'unaffected_original_mask': unaffected['original_mask'],
        'dist': min(dist0, dist180),
        'orientation': orientation,
        'com_aff': affected['original_com'],
        'com_unaff_trans': com_trans,
        'com_unaff_original': unaffected['original_com'],
        'aff_img_info': affected['image_info'],
        'unaff_img_info': unaffected['image_info'],
        'affected_laterality': affected['laterality'],
        'unaffected_laterality': unaffected['laterality'],
        # Axis endpoints
        'aff_major_axis': affected['major_axis_ends'],
        'unaff_major_axis': unaffected['major_axis_ends'],
        'trans_unaff_major_axis': (trans_major_end1, trans_major_end2),
        'aff_minor_axis': affected['minor_axis_ends'],
        'trans_unaff_minor_axis': trans_unaff_minor_ends
    }

    # Visualization
    if visualize and output_folder:
        visualize_results(results, pair, image_folder, output_folder)

    return results

def visualize_results(results, pair, image_folder, output_folder):
    '''
    UPDATED: Visualizes results with medial/lateral points as major axis endpoints
             and minor axis stored in results dictionary.
    '''
    # Load images
    aff_img = np.array(Image.open(
        os.path.join(image_folder, results['aff_img_info']['file_name'])
    ).convert('L'))

    unaff_img = np.array(Image.open(
        os.path.join(image_folder, results['unaff_img_info']['file_name'])
    ).convert('L'))

    # Setup figure - 1 row x 3 columns
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Patient {pair['patient_id']} - {pair['timepoint']} | Dist: {results['dist']:.1f}px", fontsize=16)

    # Extract axis endpoints
    aff_major_end1, aff_major_end2 = results['aff_major_axis']
    unaff_major_end1, unaff_major_end2 = results['unaff_major_axis']
    trans_major_end1, trans_major_end2 = results['trans_unaff_major_axis']
    aff_minor_end1, aff_minor_end2 = results['aff_minor_axis']
    unaff_minor_end1, unaff_minor_end2 = results['trans_unaff_minor_axis']

    # 1. Affected Head
    ax[0].imshow(aff_img, cmap='gray')
    ax[0].contour(results['affected_mask'], colors='red', linewidths=1)
    # Draw major axis (medial-lateral pillars)
    ax[0].plot([aff_major_end1[0], aff_major_end2[0]], [aff_major_end1[1], aff_major_end2[1]],
               'white', linewidth=2, alpha=0.8, label='Major Axis')
    # Plot lateral-most points as blue dots
    ax[0].scatter(
        [aff_major_end1[0], aff_major_end2[0]],
        [aff_major_end1[1], aff_major_end2[1]],
        s=20, c='blue', marker='o', label='Lateral-Most Points'
    )
    ax[0].scatter(results['com_aff'][0], results['com_aff'][1], s=40, c='white', marker='x', label='COM')
    ax[0].set_title(f"Affected Head ({results['affected_laterality']})")
    ax[0].legend(loc='upper right')
    ax[0].axis('off')

    # 2. Unaffected Head
    ax[1].imshow(unaff_img, cmap='gray')
    ax[1].contour(results['unaffected_original_mask'], colors='lime', linewidths=1)
    # Draw major axis
    ax[1].plot([unaff_major_end1[0], unaff_major_end2[0]], [unaff_major_end1[1], unaff_major_end2[1]],
               'white', linewidth=2, alpha=0.8, label='Major Axis')
    # Plot lateral-most points as blue dots
    ax[1].scatter(
        [unaff_major_end1[0], unaff_major_end2[0]],
        [unaff_major_end1[1], unaff_major_end2[1]],
        s=20, c='blue', marker='o', label='Lateral-Most Points'
    )
    ax[1].scatter(results['com_unaff_original'][0], results['com_unaff_original'][1],
                  s=40, c='white', marker='x', label='COM')
    ax[1].set_title(f"Unaffected Head ({results['unaffected_laterality']})")
    ax[1].legend(loc='upper right')
    ax[1].axis('off')

    # 3. Aligned Overlay (simplified)
    ax[2].imshow(aff_img, cmap='gray')
    ax[2].contour(results['affected_mask'], colors='red', linewidths=1, label='affected head')
    ax[2].contour(results['transformed_unaff_mask'], colors='lime', linewidths=1, linestyles='dashed', label='flipped unaffected head')
    ax[2].scatter(results['com_aff'][0], results['com_aff'][1], s=40, c='red', marker='.')
    ax[2].scatter(results['com_unaff_trans'][0], results['com_unaff_trans'][1], s=40, c ='lime', marker='.')
    ax[2].set_title('Aligned Overlay')

    ax[2].axis('off')

    plt.tight_layout()

    # Save visualization
    viz_path = os.path.join(output_folder, f"Patient_{pair['patient_id']}_{pair['timepoint']}_alignment.png")
    plt.savefig(viz_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def process_all_femoral_heads(coco_json_path, image_folder, output_folder=None,
                              visualize=False, max_pairs=None):
    '''
    Processes all femoral head pairs in a COCO dataset.
    '''
    # Find pairs with limit
    pairs = find_femoral_head_pairs(coco_json_path, max_pairs=max_pairs)

    # Load COCO data once
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Process pairs
    results = []
    total_pairs = len(pairs)
    start_time = time.time()

    for i, pair in enumerate(pairs):
        pair_start = time.time()
        res = process_femoral_head_pair(
            pair, coco_data, image_folder,
            visualize=visualize,
            output_folder=output_folder
        )

        if res:
            results.append(res)
            elapsed = time.time() - pair_start
            print(f"Processed {pair['patient_id']}-{pair['timepoint']} ({i + 1}/{total_pairs}) in {elapsed:.2f}s")

    total_time = time.time() - start_time
    print(f"Processed {len(results)} femoral head pairs in {total_time:.2f} seconds")
    return results


# Main execution
if __name__ == "__main__":
    coco_json_path = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
    image_folder = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
    output_folder = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_DI"

    os.makedirs(output_folder, exist_ok=True)

    # Process only 10 hips with simplified visualization
    aligned_results = process_all_femoral_heads(
        coco_json_path,
        image_folder,
        output_folder=output_folder,
        visualize=True,
        max_pairs=10  # Only process 10 hips
    )