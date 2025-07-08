import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.optimize import minimize_scalar

def rotate_image_and_mask(mask, angle):
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(mask, rot_mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    return rotated

def get_rotation_to_horizontal(mask):
    coords = np.column_stack(np.nonzero(mask))
    if len(coords) < 2:
        raise ValueError("Not enough points in mask for PCA.")

    coords_mean = np.mean(coords, axis=0)
    centered = coords - coords_mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]

    # Flip if pointing downward
    if major_axis[0] > 0:
        major_axis *= -1

    angle_rad = np.arctan2(major_axis[0], major_axis[1])
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg

def restore_femoral_head_spherical(mask, visualize_steps=True):
    original_area = int(np.sum(mask))
    mask = mask.astype(np.uint8)
    plt.close('all')

    angle = get_rotation_to_horizontal(mask)
    rotated = rotate_image_and_mask(mask, angle)

    if visualize_steps:
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.imshow(mask, cmap='gray')
        plt.title("Original")
        plt.subplot(1, 4, 2)
        plt.imshow(rotated, cmap='gray')
        plt.title("Rotated Horizontal")

    props = regionprops(label(rotated.astype(np.uint8)))
    if not props:
        raise ValueError("No region found in rotated mask.")

    minr, minc, maxr, maxc = props[0].bbox
    mid_y = (minr + maxr) // 1.6
    lower_half = np.zeros_like(rotated)
    lower_half[mid_y:maxr, minc:maxc] = rotated[mid_y:maxr, minc:maxc]

    if visualize_steps:
        plt.subplot(1, 4, 3)
        plt.imshow(lower_half, cmap='gray')
        plt.title("Lower Half")

    cut_line_y = mid_y
    cut_line_indices = np.argwhere(lower_half[cut_line_y, :] > 0)
    if cut_line_indices.size == 0:
        raise ValueError("No nonzero pixels found at cut line.")

    x_left = cut_line_indices.min()
    x_right = cut_line_indices.max()
    y_top = cut_line_y

    def compute_area_error(height):
        top_mask = np.zeros_like(mask)
        for x in range(x_left, x_right + 1):
            norm_x = (x - x_left) / (x_right - x_left)
            y_curve = int(y_top - height * 4 * norm_x * (1 - norm_x))
            y_curve = max(0, y_curve)
            top_mask[y_curve:y_top, x] = 1
        candidate = np.clip(lower_half + top_mask, 0, 1)
        return abs(np.sum(candidate) - original_area)

    result = minimize_scalar(compute_area_error, bounds=(10, 300), method='bounded')
    optimal_height = result.x
    top_mask = np.zeros_like(mask)

    for x in range(x_left, x_right + 1):
        norm_x = (x - x_left) / (x_right - x_left)
        y_curve = int(y_top - optimal_height * 4 * norm_x * (1 - norm_x))
        y_curve = max(0, y_curve)
        top_mask[y_curve:y_top, x] = 1

    candidate = np.clip(lower_half + top_mask, 0, 1)

    if visualize_steps:
        plt.subplot(1, 4, 4)
        plt.imshow(candidate, cmap='gray')
        plt.title("Restored in Rotated Space")
        plt.tight_layout()
        plt.show()

    restored = rotate_image_and_mask(candidate, -angle)
    return (restored > 0).astype(np.uint8)
