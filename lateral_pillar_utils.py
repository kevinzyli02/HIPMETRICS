import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from lateral_pillar_reference import compute_lateral_pillar_reference
from lateral_pillar_reference import overlay_hemisphere_on_mask


def compute_lateral_pillar(mask: np.ndarray, filename: str, visualize: bool = False):
    """
    Compute lateral pillar height ratio and assign Herring classification.
    Rotate femoral head mask to upright orientation before measuring.

    Parameters:
        mask (np.ndarray): 2D binary mask of the femoral head
        filename (str): Image filename containing 'L' or 'R' to determine laterality
        visualize (bool): If True, displays the mask and lateral third region

    Returns:
        dict: {
            'lateral_height': int,
            'epiphyseal_height': int,
            'lateral_ratio': float,
            'herring_class': str
        }
    """

    # Check for empty mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return {
            'lateral_height': 0,
            'epiphyseal_height': 0,
            'lateral_ratio': 0.0,
            'herring_class': 'Unknown'
        }

    # === Step 1: Compute PCA to find orientation angle ===
    coords = np.column_stack((xs, ys))
    pca = PCA(n_components=2)
    pca.fit(coords)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])  # radians

    # Convert to degrees and adjust so major axis points vertically (90 degrees)
    angle_deg = np.degrees(angle)
    rotate_deg = 90 - angle_deg

    # === Step 2: Rotate mask to upright ===
    # scipy.ndimage.rotate rotates counter-clockwise by default
    rotated_mask = rotate(mask, rotate_deg, reshape=True, order=0)  # order=0 nearest-neighbor for masks

    # Update mask to rotated version
    mask = (rotated_mask > 0.5).astype(np.uint8)  # binarize

    # === Step 3: Bounding box of rotated mask ===
    ys, xs = np.where(mask > 0)
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    # === Step 4: Determine laterality ===
    filename = filename.upper()
    is_left_hip = 'L' in filename
    is_right_hip = 'R' in filename

    if not (is_left_hip or is_right_hip):
        raise ValueError(f"Cannot determine laterality from filename: {filename}")

    # === Step 5: Divide into thirds ===
    width = x_max - x_min
    third_width = width // 3

    if is_right_hip:
        lat_x_min = x_min
        lat_x_max = x_min + third_width
    else:  # left hip
        lat_x_min = x_max - third_width
        lat_x_max = x_max

    # === Step 6: Extract lateral mask and compute height ===
    lateral_mask = mask[:, lat_x_min:lat_x_max]
    ys_lateral, _ = np.where(lateral_mask > 0)

    lateral_height = np.max(ys_lateral) - np.min(ys_lateral) if len(ys_lateral) > 0 else 0
    epiphyseal_height = y_max - y_min if y_max > y_min else 1  # prevent divide-by-zero

    lateral_ratio = lateral_height / epiphyseal_height

    # === Step 7: Herring classification based on ratio ===
    if lateral_ratio > 0.5:
        herring_class = 'A'
    elif 0.5 >= lateral_ratio >= 0.33:
        herring_class = 'B'
    else:
        herring_class = 'C'

    # === Step 8: modified pillar reference

    #reference = compute_lateral_pillar_reference(mask, visualize=True)
    # === Step 9: modified pillar reference

    #overlay = overlay_hemisphere_on_mask(mask, reference['radius'], reference['lateral_third_height'],
                                         #lateral_height, visualize=false)
    if visualize:
        plt.imshow(mask, cmap='gray')
        plt.axvline(lat_x_min, color='blue', linestyle='--')
        plt.axvline(lat_x_max, color='blue', linestyle='--')
        plt.title(f'Lateral Pillar: {lateral_ratio:.2f} ({herring_class})')
        plt.axis('off')
        plt.show()



    return {
        'lateral_height': lateral_height,
        'epiphyseal_height': epiphyseal_height,
        'lateral_ratio': lateral_ratio,
        'herring_class': herring_class
    }
