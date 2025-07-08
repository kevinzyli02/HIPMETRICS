import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from skimage import measure
from sklearn.decomposition import PCA
import os
import shutil

def compute_lateral_pillar(mask: np.ndarray, filename: str, image: np.ndarray = None, visualize: bool = False, restored_mask: np.ndarray = None, output_dir: str = None):
    filename = filename.upper()
    is_left = 'L' in filename
    is_right = 'R' in filename
    if not (is_left or is_right):
        raise ValueError(f"Cannot determine laterality from filename: {filename}")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return {
            'actual_lateral_height': 0,
            'restored_lateral_height': 0,
            'pillar_ratio': 0.0
        }

    coords = np.column_stack((xs, ys))
    pca = PCA(n_components=2)
    pca.fit(coords)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

    center = np.mean(coords, axis=0)
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    coords_centered = coords - center
    rotated_coords = coords_centered @ rotation_matrix.T

    x_rot = rotated_coords[:, 0]
    y_rot = rotated_coords[:, 1]
    x_min, x_max = np.min(x_rot), np.max(x_rot)
    width = x_max - x_min
    third_width = width / 3

    if is_left:
        lat_x_min = x_max - third_width
        lat_x_max = x_max
    else:
        lat_x_min = x_min
        lat_x_max = x_min + third_width

    lateral_mask_indices = (x_rot >= lat_x_min) & (x_rot < lat_x_max)
    y_lateral = y_rot[lateral_mask_indices]
    actual_lateral_height = np.max(y_lateral) - np.min(y_lateral) if len(y_lateral) > 0 else 0

    if restored_mask is not None:
        ys_restored, xs_restored = np.where(restored_mask > 0)
        coords_restored = np.column_stack((xs_restored, ys_restored))
        coords_restored_centered = coords_restored - center
        rotated_coords_restored = coords_restored_centered @ rotation_matrix.T

        x_rot_rest = rotated_coords_restored[:, 0]
        y_rot_rest = rotated_coords_restored[:, 1]

        lateral_rest_indices = (x_rot_rest >= lat_x_min) & (x_rot_rest < lat_x_max)
        y_lateral_rest = y_rot_rest[lateral_rest_indices]
        restored_lateral_height = np.max(y_lateral_rest) - np.min(y_lateral_rest) if len(y_lateral_rest) > 0 else 1
    else:
        restored_lateral_height = 1

    pillar_ratio = actual_lateral_height / restored_lateral_height if restored_lateral_height > 0 else 0.0

    if visualize and image is not None:
        plt.figure(figsize=(6, 8), dpi=300)
        ax = plt.gca()
        ax.imshow(image, cmap='gray')

        contours_actual = measure.find_contours(mask, 0.5)
        for contour in contours_actual:
            ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)

        if restored_mask is not None:
            contours_restored = measure.find_contours(restored_mask, 0.5)
            for contour in contours_restored:
                ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

        thirds_x = [x_min + third_width, x_min + 2 * third_width]
        for x in thirds_x:
            p1 = np.array([x, -100])
            p2 = np.array([x, 100])
            p1_global = (p1 @ rotation_matrix) + center
            p2_global = (p2 @ rotation_matrix) + center
            ax.plot([p1_global[0], p2_global[0]], [p1_global[1], p2_global[1]], linestyle='--', color='cyan', linewidth=1.2)

        # Draw bounding box over lateral third where height is measured
        bbox_x = lat_x_min
        bbox_width = third_width
        bbox_y = np.min(y_lateral) if len(y_lateral) > 0 else 0
        bbox_height = actual_lateral_height
        bbox_origin = (bbox_x, bbox_y)
        bbox_coords = np.array([
            [bbox_x, bbox_y],
            [bbox_x + bbox_width, bbox_y],
            [bbox_x + bbox_width, bbox_y + bbox_height],
            [bbox_x, bbox_y + bbox_height]
        ])
        bbox_coords_global = (bbox_coords @ rotation_matrix) + center
        patch = plt.Polygon(bbox_coords_global, closed=True, edgecolor='yellow', fill=False, linewidth=2)
        ax.add_patch(patch)

        ax.set_title(f"{filename} | Lateral Pillar Ratio: {pillar_ratio:.2f}", color='black')
        legend_elements = [
            Patch(facecolor='white', edgecolor='white', label='Actual Mask'),
            Patch(facecolor='red', edgecolor='red', label='Restored Mask'),
            Patch(facecolor='cyan', edgecolor='cyan', label='Lateral Thirds', linestyle='--'),
            Patch(facecolor='none', edgecolor='yellow', label='Height Measurement Box')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.axis('off')
        ax.set_facecolor('black')
        plt.tight_layout()

        subfolder = "frog" if "FROG" in filename else "ap"
        grade = "A" if 0.96 <= pillar_ratio <= 1.2 else "B" if 0.8 <= pillar_ratio < 0.96 else "C"
        output_dir_final = os.path.join(output_dir, subfolder, grade)
        os.makedirs(output_dir_final, exist_ok=True)

        save_path = os.path.join(output_dir_final, f"{filename}_overlay.jpg")
        plt.savefig(save_path, format='jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

    return {
        'actual_lateral_height': actual_lateral_height,
        'restored_lateral_height': restored_lateral_height,
        'pillar_ratio': pillar_ratio
    }
