import numpy as np
import matplotlib.pyplot as plt

def compute_lateral_pillar_reference(mask: np.ndarray, visualize: bool = True):
    """
    Compute reference lateral pillar height from a hemisphere approximating the femoral head.

    Parameters:
        mask (np.ndarray): 2D binary mask of the femoral head.
        visualize (bool): Whether to plot the hemisphere cross-section and highlight the lateral third.

    Returns:
        dict: {
            'volume': int,
            'radius': float,
            'lateral_third_height': float
        }
    """
    # Rotate mask so short axis is vertical
    mask = rotate_mask_to_short_axis(mask)

    # Step 1: Compute the "volume" (area in 2D) of the femoral head mask
    volume_pixels = np.sum(mask)

    # Step 2: Approximate the femoral head as a hemisphere
    # Volume of a hemisphere = (2/3) * π * r^3
    # Solve for r: r = (3*V / (2π))^(1/3)
    r = (2 * volume_pixels / (np.pi)) ** (1 / 2)

    # Step 3: Find height where top 1/3 volume begins in hemisphere
    # Volume from base up to height h in hemisphere: V(h) = (πh^2/3)(3R - h)
    # Goal: V(h_cutoff) = 1/3 * total volume
    # We solve this numerically
    R = r
    h = (8/9)**( 1/2) * R

    #hs = np.linspace(0, R, 1000)
    #volumes = (np.pi * hs ** 2 / 3) * (3 * R - hs)

    #idx = np.where(volumes >= target_volume)[0][0]
    #h_cutoff = hs[idx]  # height above which the top 1/3 lies



    return {
        'volume': volume_pixels,
        'radius': R,
        'lateral_third_height': h  # height of top third
    }


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import center_of_mass

def overlay_hemisphere_on_mask(mask, radius, lateral_third_height, lateral_height, visualize=True):
    """
    Overlay a hemisphere with its center aligned to the center of mass of the femoral head mask.

    Args:
        mask (np.ndarray): Binary mask of the femoral head.
        radius (float): Radius of the hemisphere (based on volume).
        lateral_third_height (float): Equivalent hemisphere height.
        lateral_height(float): Actual lateral height of femoral head
        visualize (bool): Whether to plot the overlay.

    Returns:
        np.ndarray: The binary hemisphere mask aligned with the femoral head mask.
    """
    # Ensure mask is 2D uint8
    mask = rotate_mask_to_short_axis(mask)
    mask = np.squeeze(mask)
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    mask = np.ascontiguousarray(mask)

    canvas_height, canvas_width = mask.shape

    # Compute center of mass
    com_y, com_x = center_of_mass(mask)
    com_x = int(com_x)

    ys, xs = np.where(mask > 0)
    base_y = int(np.max(ys))  # bottom of femoral head

    # Compute Y position for hemisphere center (so base lines up with base_y)
    center_y = base_y

    # Create blank hemisphere canvas
    hemisphere_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Draw filled circle centered at center of mass
    cv2.circle(hemisphere_mask, (com_x, center_y), int(radius), color=1, thickness=-1)

    # Mask out lower half to create upper hemisphere
    hemisphere_mask[center_y + 1:, :] = 0

    #Compute lateral ratio = lateral height / lateral third height
    lateral_ratio = lateral_height / lateral_third_height
    if visualize:

        plt.figure(figsize=(8, 8), dpi=300)  # Increase DPI and figure size
        plt.imshow(mask, cmap='gray', alpha=0.6)
        plt.imshow(hemisphere_mask, cmap='Blues', alpha=0.6)
        #plt.scatter([com_x], [com_y], color='red', label='Center of Mass')
        plt.title(f'Lateral Pillar: {lateral_ratio:.2f} ')
        plt.axis('off')
        plt.legend()
        plt.show()

    return hemisphere_mask


from sklearn.decomposition import PCA
from scipy.ndimage import rotate


def rotate_mask_to_short_axis(mask):
    """Rotate binary mask so the short PCA axis is vertical (aligned with image Y-axis)."""
    ys, xs = np.where(mask > 0)
    coords = np.column_stack((xs, ys))
    pca = PCA(n_components=2)
    pca.fit(coords)

    # Short axis = second principal component (less variance)
    angle = np.arctan2(pca.components_[1, 1], pca.components_[1, 0])  # radians
    angle_deg = np.degrees(angle)

    # Rotate mask so short axis is vertical (aligned with Y)
    rotation_needed = 360 - angle_deg
    rotated = rotate(mask, rotation_needed, reshape=True, order=0)

    # Return binary
    return (rotated > 0.5).astype(np.uint8)


def rotate_mask_horizontal(mask):
    """
    Rotate the femoral head mask so its principal axis is aligned horizontally.

    Args:
        mask (np.ndarray): Binary femoral head mask.

    Returns:
        np.ndarray: Rotated mask.
        float: Angle in degrees used for rotation.
    """
    # Find coordinates of the mask
    ys, xs = np.where(mask > 0)
    coords = np.column_stack((xs, ys))

    # PCA to find angle of the long axis
    pca = PCA(n_components=2)
    pca.fit(coords)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])  # radians
    angle_deg = np.degrees(angle)

    # Rotate to align long axis horizontally
    rotated = rotate(mask, angle=-angle_deg, reshape=True, order=0)  # counter-clockwise
    rotated = (rotated > 0.5).astype(np.uint8)  # binarize

    return rotated, angle_deg