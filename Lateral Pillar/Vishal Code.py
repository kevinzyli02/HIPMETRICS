import os
import cv2
import numpy as np
import glob
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize
from sklearn.linear_model import LinearRegression

# Paths
radiographs_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\split_dataset\herringAP1'
gt_masks_path = os.path.join(radiographs_path, 'masks', 'gt')
head_masks_path = os.path.join(radiographs_path, 'masks', 'head')
shaft_masks_path = os.path.join(radiographs_path, 'masks', 'shaft')
wholeshaft_masks_path = os.path.join(radiographs_path, 'masks', 'wholeshaft')
acetabulum_masks_path = os.path.join(radiographs_path, 'masks', 'acetabulum')

output_path = os.path.join(radiographs_path, 'annotated_radiographs')

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)


# Function to find the superiormost point in the mask
def find_superiormost_point(mask):
    try:
        indices = np.where(mask == 255)
        if indices[0].size == 0:
            return None
        y_min = np.min(indices[0])
        x_at_y_min = indices[1][np.argmin(indices[0])]
        return (x_at_y_min, y_min)
    except Exception as e:
        print(f"Error in find_superiormost_point: {e}")
        return None


# Function to find the lateral and medial points using intersection
import os
import cv2
import numpy as np


def find_and_draw_medial_lateral_points(head_mask, annotated_radiograph, base_name):
    try:
        # Find contours of the head mask
        contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the head mask.")

        # Get the largest contour (assuming it's the head)
        contour = max(contours, key=cv2.contourArea)

        # Find the two points with maximum distance
        distances = []
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                dist = np.linalg.norm(contour[i] - contour[j])
                distances.append((dist, contour[i][0], contour[j][0]))

        # Get the pair of points with maximum distance
        max_dist, point1, point2 = max(distances, key=lambda x: x[0])

        # Draw points on the annotated radiograph
        cv2.circle(annotated_radiograph, tuple(point1), 5, (255, 255, 0), -1)  # Blue
        cv2.circle(annotated_radiograph, tuple(point2), 5, (255, 255, 0), -1)  # Blue

        return annotated_radiograph

    except Exception as e:
        print(f"Error in find_and_draw_medial_lateral_points for {base_name}: {e}")
        return annotated_radiograph


# Function to model shaft as quadrangle and find axes
def model_shaft_and_find_axes(shaft_mask):
    try:
        contours, _ = cv2.findContours(shaft_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        contour = max(contours, key=cv2.contourArea)
        hull = ConvexHull(contour.squeeze())
        hull_points = contour.squeeze()[hull.vertices]

        rect = cv2.minAreaRect(hull_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Sort points to get consistent order
        box = box[np.argsort(box[:, 1])]
        top_points = box[:2]
        bottom_points = box[2:]

        top_points = top_points[np.argsort(top_points[:, 0])]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]

        # Calculate midpoints
        top_mid = np.mean(top_points, axis=0).astype(int)
        bottom_mid = np.mean(bottom_points, axis=0).astype(int)
        left_mid = np.mean([top_points[0], bottom_points[0]], axis=0).astype(int)
        right_mid = np.mean([top_points[1], bottom_points[1]], axis=0).astype(int)

        # Calculate axes
        major_axis = (top_mid, bottom_mid)
        minor_axis = (left_mid, right_mid)

        # Calculate angles
        major_angle = np.abs(90 - np.abs(np.rad2deg(np.arctan2(major_axis[1][1] - major_axis[0][1],
                                                               major_axis[1][0] - major_axis[0][0]))))
        minor_angle = np.abs(90 - np.abs(np.rad2deg(np.arctan2(minor_axis[1][1] - minor_axis[0][1],
                                                               minor_axis[1][0] - minor_axis[0][0]))))

        print(f"Major axis angle: {major_angle:.2f}, Minor axis angle: {minor_angle:.2f}")

        if major_angle < minor_angle:
            return major_axis, major_angle, "major"
        else:
            return minor_axis, minor_angle, "minor"
    except Exception as e:
        print(f"Error in model_shaft_and_find_axes: {e}")
        return None, None, None


# Function to draw a line on the image
def draw_line(image, start_point, end_point, color=(128, 128, 128), thickness=2):
    cv2.line(image, tuple(start_point), tuple(end_point), color, thickness)
    return image


# Function to draw points along the line
def draw_points_along_line(image, start_point, end_point, color=(128, 0, 128)):
    points = [
        tuple(map(int, start_point + (end_point - start_point) * 1 / 3)),
        tuple(map(int, start_point + (end_point - start_point) * 1 / 2)),
        tuple(map(int, start_point + (end_point - start_point) * 2 / 3))
    ]
    for point in points:
        cv2.circle(image, point, 5, color, -1)
    return image


# Processing each mask
for gt_mask_file in glob.glob(os.path.join(gt_masks_path, '*.bmp')):
    try:
        base_name = os.path.basename(gt_mask_file)
        print(f"Processing {base_name}")

        # Read the gt mask and the corresponding radiograph
        gt_mask = cv2.imread(gt_mask_file, cv2.IMREAD_GRAYSCALE)
        radiograph_file = os.path.join(radiographs_path, base_name)
        radiograph = cv2.imread(radiograph_file)

        if gt_mask is None or radiograph is None:
            print(f"Error reading files: {gt_mask_file}, {radiograph_file}")
            continue

        # Create a copy of the radiograph for annotation
        annotated_radiograph = radiograph.copy()

        # Find the superiormost point of the gt mask
        gt_superiormost_point = find_superiormost_point(gt_mask)
        if gt_superiormost_point:
            cv2.circle(annotated_radiograph, gt_superiormost_point, 5, (139, 0, 0), -1)  # Dark blue

        # Read the head mask
        head_mask_file = os.path.join(head_masks_path, base_name)
        head_mask = cv2.imread(head_mask_file, cv2.IMREAD_GRAYSCALE)

        if head_mask is None:
            print(f"Error reading head mask: {head_mask_file}")
            continue

        # Find the superiormost point of the head mask
        head_superiormost_point = find_superiormost_point(head_mask)
        if head_superiormost_point:
            cv2.circle(annotated_radiograph, head_superiormost_point, 5, (0, 0, 255), -1)  # Red

        # Determine if the image is left or right
        is_left_side = base_name.endswith('_L.bmp')

        # Read the wholeshaft mask
        wholeshaft_mask_file = os.path.join(wholeshaft_masks_path, base_name)
        wholeshaft_mask = cv2.imread(wholeshaft_mask_file, cv2.IMREAD_GRAYSCALE)

        if wholeshaft_mask is None:
            print(f"Error reading wholeshaft mask: {wholeshaft_mask_file}")
            continue

        # Find and draw medial and lateral points
        annotated_radiograph = find_and_draw_medial_lateral_points(head_mask, annotated_radiograph, base_name)

        # Read the shaft mask
        shaft_mask_file = os.path.join(shaft_masks_path, base_name)
        shaft_mask = cv2.imread(shaft_mask_file, cv2.IMREAD_GRAYSCALE)

        if shaft_mask is None:
            print(f"Error reading shaft mask: {shaft_mask_file}")
            continue

        # Model shaft and find axes
        axis, angle, axis_type = model_shaft_and_find_axes(shaft_mask)

        if axis is not None:
            print(f"Using {axis_type} axis with angle {angle:.2f}")

            # Draw the axis on the radiograph
            annotated_radiograph = draw_line(annotated_radiograph, axis[0], axis[1], color=(216, 191, 216), thickness=2)

            # Draw the points along the axis
            annotated_radiograph = draw_points_along_line(annotated_radiograph, np.array(axis[0]), np.array(axis[1]),
                                                          color=(128, 0, 128))

        # Save the annotated radiograph
        output_file = os.path.join(output_path, base_name)
        cv2.imwrite(output_file, annotated_radiograph)
        print(f"Saved annotated radiograph: {output_file}")

    except Exception as e:
        print(f"Error processing {base_name}: {e}")

print("Processing complete.")

# EQ

import os
import csv
import cv2
import numpy as np

# Define directories
radiographs_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\train\herringAP2'
head_masks_path = os.path.join(radiographs_path, 'masks', 'head')
annotated_radiographs = os.path.join(radiographs_path, 'annotated_radiographs')
EQ_directory = os.path.join(annotated_radiographs, 'EQ')

# Create output directory
os.makedirs(EQ_directory, exist_ok=True)

# CSV file path
output_csv_path = os.path.join(annotated_radiographs, 'EQ_radiograph_measurements.csv')


def find_blue_points(image):
    """Find blue markers (BGR: 0, 255, 255)"""
    blue_mask = cv2.inRange(image, (255, 255, 0), (255, 255, 0))
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
    return centroids


def calculate_perpendicular_line(midpoint, direction, image_shape):
    """Calculate perpendicular line across entire image"""
    perp_vector = np.array([-direction[1], direction[0]], dtype=np.float32)
    perp_vector /= np.linalg.norm(perp_vector)

    # Extend line to image edges
    t_values = [
        (0 - midpoint[0]) / perp_vector[0],
        (image_shape[1] - 1 - midpoint[0]) / perp_vector[0],
        (0 - midpoint[1]) / perp_vector[1],
        (image_shape[0] - 1 - midpoint[1]) / perp_vector[1]
    ]

    t_min, t_max = max(min(t for t in t_values if t >= 0), -1000), min(max(t for t in t_values if t <= 0), 1000)

    start_point = (int(midpoint[0] + perp_vector[0] * t_min), int(midpoint[1] + perp_vector[1] * t_min))
    end_point = (int(midpoint[0] + perp_vector[0] * t_max), int(midpoint[1] + perp_vector[1] * t_max))

    return start_point, end_point


def calculate_height_line(start, end, mask):
    """Calculate height line within femoral head mask"""
    height = 0
    height_points = []

    # Use Bresenham's line algorithm
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < mask.shape[1] and 0 <= y0 < mask.shape[0]:
            if mask[y0, x0] == 255:
                height += 1
                height_points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return height, height_points


def process_image(file_path, mask_path, output_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read image: {file_path}")
        return None, None, None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return None, None, None

    centroids = find_blue_points(image)
    if len(centroids) != 2:
        print(f"Found {len(centroids)} blue markers in {file_path}")
        return None, None, None

    pt1, pt2 = np.array(centroids[0]), np.array(centroids[1])
    width = int(np.linalg.norm(pt1 - pt2))
    cv2.line(image, tuple(pt1), tuple(pt2), (250, 200, 0), 2)

    midpoint = ((pt1 + pt2) // 2).astype(int)
    direction = pt2 - pt1

    perp_start, perp_end = calculate_perpendicular_line(midpoint, direction, image.shape)
    cv2.line(image, perp_start, perp_end, (0, 200, 255), 2)  # Blue perpendicular line

    height, height_points = calculate_height_line(perp_start, perp_end, mask)

    # Draw red height line
    for i in range(len(height_points) - 1):
        cv2.line(image, height_points[i], height_points[i + 1], (0, 0, 255), 2)

    cv2.imwrite(output_path, image)
    return width, height, width / height if height else 0


# Initialize CSV
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Sample Name', 'Width', 'Height', 'EQ'])

# Process files
for file_name in os.listdir(annotated_radiographs):
    if file_name.lower().endswith('.bmp'):
        sample_name = os.path.splitext(file_name)[0]
        print(f"Processing {sample_name}...")

        bmp_path = os.path.join(annotated_radiographs, file_name)
        mask_path = os.path.join(head_masks_path, f"{sample_name}.bmp")

        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue

        output_path = os.path.join(EQ_directory, file_name)
        width, height, eq = process_image(bmp_path, mask_path, output_path)

        if width is not None and height is not None:
            with open(output_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([sample_name, width, height, eq])
            print(f"Processed {sample_name}: W={width}, H={height}, EQ={eq:.2f}")

print("Processing complete. Check EQ directory for annotated images.")

# Lateral pillar


import os
import csv
import cv2
import numpy as np

# Define directories
radiographs_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\train\herringAP2'
head_masks_path = os.path.join(radiographs_path, 'masks', 'head')
annotated_radiographs = os.path.join(radiographs_path, 'annotated_radiographs')
lateral_pillar_directory = os.path.join(annotated_radiographs, 'lateral pillar')

# Create output directory
os.makedirs(lateral_pillar_directory, exist_ok=True)

# CSV file path
output_csv_path = os.path.join(annotated_radiographs, 'lateralpillar_radiograph_measurements.csv')


def find_blue_points(image):
    """Find blue markers (BGR: 255, 255, 0)"""
    blue_mask = cv2.inRange(image, (255, 255, 0), (255, 255, 0))
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
    return centroids


def calculate_perpendicular_line(point, direction, image_shape):
    """Calculate perpendicular line across entire image"""
    perp_vector = np.array([-direction[1], direction[0]], dtype=np.float32)
    perp_vector /= np.linalg.norm(perp_vector)

    t_values = [
        (0 - point[0]) / perp_vector[0],
        (image_shape[1] - 1 - point[0]) / perp_vector[0],
        (0 - point[1]) / perp_vector[1],
        (image_shape[0] - 1 - point[1]) / perp_vector[1]
    ]

    t_min, t_max = max(min(t for t in t_values if t >= 0), -1000), min(max(t for t in t_values if t <= 0), 1000)

    start_point = (int(point[0] + perp_vector[0] * t_min), int(point[1] + perp_vector[1] * t_min))
    end_point = (int(point[0] + perp_vector[0] * t_max), int(point[1] + perp_vector[1] * t_max))

    return start_point, end_point


def calculate_line_in_mask(start, end, mask):
    """Calculate line length within femoral head mask"""
    length = 0
    points = []

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < mask.shape[1] and 0 <= y0 < mask.shape[0]:
            if mask[y0, x0] == 255:
                length += 1
                points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return length, points


def process_image(file_path, mask_path, output_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Could not read image: {file_path}")
        return None, None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return None, None

    centroids = find_blue_points(image)
    if len(centroids) != 2:
        print(f"Found {len(centroids)} blue markers in {file_path}")
        return None, None

    pt1, pt2 = np.array(centroids[0]), np.array(centroids[1])
    cv2.line(image, tuple(pt1), tuple(pt2), (250, 200, 0), 2)  # Width line

    direction = pt2 - pt1
    one_third = pt1 + direction * (1 / 3)
    two_thirds = pt1 + direction * (2 / 3)

    perp_start_1, perp_end_1 = calculate_perpendicular_line(one_third, direction, image.shape)
    perp_start_2, perp_end_2 = calculate_perpendicular_line(two_thirds, direction, image.shape)

    height_1, points_1 = calculate_line_in_mask(perp_start_1, perp_end_1, mask)
    height_2, points_2 = calculate_line_in_mask(perp_start_2, perp_end_2, mask)

    # Draw both lines in red
    for i in range(len(points_1) - 1):
        cv2.line(image, points_1[i], points_1[i + 1], (0, 0, 255), 2)
    for i in range(len(points_2) - 1):
        cv2.line(image, points_2[i], points_2[i + 1], (0, 0, 255), 2)

    cv2.imwrite(output_path, image)

    return height_1, height_2


# Initialize CSV
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Sample Name', 'First Line Length', 'Second Line Length'])

# Process files
for file_name in os.listdir(annotated_radiographs):
    if file_name.lower().endswith('.bmp'):
        sample_name = os.path.splitext(file_name)[0]
        print(f"Processing {sample_name}...")

        bmp_path = os.path.join(annotated_radiographs, file_name)
        mask_path = os.path.join(head_masks_path, file_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue

        output_path = os.path.join(lateral_pillar_directory, file_name)
        first_length, second_length = process_image(bmp_path, mask_path, output_path)

        if first_length is not None and second_length is not None:
            with open(output_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([sample_name, first_length, second_length])
            print(f"Processed {sample_name}: First Length={first_length}, Second Length={second_length}")

print("Processing complete. Check 'lateral pillar' directory for annotated images.")

# ATD
import os
import cv2
import numpy as np
import csv

# Define directories
radiographs_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\train\herringAP2'
annotated_radiographs = os.path.join(radiographs_path, 'annotated_radiographs')
ATD_directory = os.path.join(annotated_radiographs, 'ATD')
# Create output directory
os.makedirs(ATD_directory, exist_ok=True)


def find_centroids(image, color):
    mask = cv2.inRange(image, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids


def extend_line(p1, p2, img_shape):
    height, width = img_shape[:2]
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')

    if slope == float('inf'):
        return [(p1[0], 0), (p1[0], height - 1)]

    y_intercept = p1[1] - slope * p1[0]

    x1, y1 = 0, int(y_intercept)
    x2, y2 = width - 1, int(slope * (width - 1) + y_intercept)

    return [(x1, y1), (x2, y2)]


def distance_point_to_line(point, line_point1, line_point2):
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def intersection_point(point, line_point1, line_point2):
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = (dy * (y0 - y1) + dx * (x0 - x1)) / det
    return int(x1 + a * dx), int(y1 + a * dy)


measurements = []

for filename in os.listdir(annotated_radiographs):
    if filename.endswith('.bmp'):
        img_path = os.path.join(annotated_radiographs, filename)
        img = cv2.imread(img_path)

        # Find purple centroids (midshaft points)
        purple_centroids = find_centroids(img, (128, 0, 128))
        if len(purple_centroids) != 3:
            print(f"Warning: Expected 3 purple points, found {len(purple_centroids)} in {filename}")
            continue

        # Draw purple line
        purple_line = extend_line(purple_centroids[0], purple_centroids[-1], img.shape)
        cv2.line(img, purple_line[0], purple_line[1], (128, 0, 128), 2)

        # Find blue and red points
        blue_centroids = find_centroids(img, (139, 0, 0))
        red_centroids = find_centroids(img, (0, 0, 255))
        if len(blue_centroids) != 1 or len(red_centroids) != 1:
            print(
                f"Warning: Expected 1 blue and 1 red point, found {len(blue_centroids)} blue and {len(red_centroids)} red in {filename}")
            continue

        blue_point = blue_centroids[0]
        red_point = red_centroids[0]

        # Draw orthogonal lines
        blue_intersection = intersection_point(blue_point, purple_line[0], purple_line[1])
        red_intersection = intersection_point(red_point, purple_line[0], purple_line[1])

        cv2.line(img, blue_point, blue_intersection, (139, 0, 0), 2)
        cv2.line(img, red_point, red_intersection, (0, 0, 255), 2)

        # Thicken purple line between intersections
        cv2.line(img, blue_intersection, red_intersection, (128, 0, 128), 4)

        # Calculate length of thickened line
        length = np.sqrt((blue_intersection[0] - red_intersection[0]) ** 2 +
                         (blue_intersection[1] - red_intersection[1]) ** 2)

        # Save annotated image
        output_path = os.path.join(ATD_directory, filename)
        cv2.imwrite(output_path, img)

        # Store measurement
        measurements.append([filename, length])

# Save measurements to CSV
csv_path = os.path.join(annotated_radiographs, 'ATD_radiograph_measurements.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Length'])
    writer.writerows(measurements)

print("Processing complete. Annotated images saved in ATD directory and measurements saved in CSV file.")