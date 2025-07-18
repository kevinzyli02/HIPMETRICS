import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rotate
import pandas as pd
import os
from pathlib import Path


class FemoralHeadAnalyzer:
    def __init__(self, result_dict, output_folder):
        """
        Initialize with a results dictionary for one patient/timepoint.

        Args:
            result_dict (dict): Dictionary containing patient data and masks
            output_folder (str): Path to save visualizations and Excel files
        """
        self.data = result_dict
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.rotated_aff_mask = None
        self.rotated_unaff_mask = None
        self.pillar_measurements = {}
        self.eq_measurements = {}
        self.deformity_index = None  # Add attribute for deformity index

    def _compute_rotation_angle(self, axis_endpoints):
        """Calculate rotation angle to make major axis horizontal."""
        (x1, y1), (x2, y2) = axis_endpoints
        dx, dy = x2 - x1, y2 - y1
        return np.degrees(np.arctan2(dy, dx))  # Negative to counter-clockwise rotation

    def _rotate_mask(self, mask, angle, center=None):
        """
        Rotate mask around a specified center.

        Args:
            mask: Binary mask to rotate
            angle: Rotation angle in degrees
            center: Custom rotation center (x, y). If None, uses center of mass.
        """
        # Use custom center if provided, else calculate center of mass
        if center is None:
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return mask
            cx, cy = np.mean(x_indices), np.mean(y_indices)
        else:
            cx, cy = center

        # Rotate mask around the specified center
        rotated = rotate(
            mask.astype(float),
            angle,
            center=(cx, cy),
            preserve_range=True,
            mode='constant',
            cval=0
        )
        return rotated > 0.5

    def _get_major_axis_center(self, axis_endpoints):
        """Calculate midpoint of major axis"""
        (x1, y1), (x2, y2) = axis_endpoints
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _rotate_point(self, point, center, angle_deg):
        """Rotate a point around a center by given angle in degrees."""
        angle_rad = np.radians(angle_deg)
        x, y = point
        cx, cy = center

        # Translate point to origin
        x_translated = x - cx
        y_translated = y - cy

        # Apply rotation
        x_rotated = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
        y_rotated = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)

        # Translate back
        new_x = x_rotated + cx
        new_y = y_rotated + cy

        return new_x, new_y

    def align_major_axis(self):
        """Rotate masks around center of major axis to make it horizontal,
        with additional 180° rotation check for unaffected mask."""
        # Calculate rotation angle
        angle = self._compute_rotation_angle(self.data['aff_major_axis'])

        # Get rotation centers
        aff_center = self._get_major_axis_center(self.data['aff_major_axis'])
        unaff_center = self._get_major_axis_center(self.data['trans_unaff_major_axis'])

        # Rotate both masks using their respective major axis centers
        self.rotated_aff_mask = self._rotate_mask(
            self.data['affected_mask'], angle, center=aff_center
        )
        self.rotated_unaff_mask = self._rotate_mask(
            self.data['transformed_unaff_mask'], angle, center=unaff_center
        )

        # --- 180° rotation check for unaffected mask ---
        # Get original center of mass for unaffected mask
        com_orig = self.data['com_unaff_trans']  # (x, y)
        com_orig_rot = self._rotate_point(com_orig, unaff_center, angle)

        # Calculate rotated COM positions
        com_rotated = self._rotate_point(com_orig_rot, unaff_center, angle)
        com_rotated_180 = self._rotate_point(com_rotated, unaff_center, 180)

        # Calculate relative COM positions
        aff_com_vector = np.array(self.data['com_aff']) - np.array(aff_center)
        unaff_com_vector = np.array(com_rotated) - np.array(unaff_center)


        # Get vertical directions (y-components)
        aff_vertical_sign = np.sign(aff_com_vector[1])
        unaff_vertical_sign0 = np.sign(unaff_com_vector[1])

        # Flip if vertical orientations don't match
        if aff_vertical_sign != unaff_vertical_sign0:
            self.rotated_unaff_mask = self._rotate_mask(
                self.rotated_unaff_mask, 180, center=unaff_center
            )
            self.rotated_aff_mask = self._rotate_mask(
                self.rotated_aff_mask, 180, center=aff_center
            )
            self.unaff_flipped = True

        else:
            self.unaff_flipped = False


        # Visualize rotation
        return self._visualize_rotation(angle, aff_center, unaff_center, com_rotated, com_rotated_180, com_orig)

    def _visualize_rotation(self, angle, aff_center, unaff_center, com_rotated,com_rotated_180, com_orig):
        """Visualize rotation centers and results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Original affected mask
        ax = axes[0, 0]
        ax.imshow(self.data['affected_mask'], cmap='gray')
        (x1, y1), (x2, y2) = self.data['aff_major_axis']
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        ax.scatter([x1, x2], [y1, y2], c=['red', 'blue'], s=50)
        ax.scatter(aff_center[0], aff_center[1], c='yellow', s=100, marker='*')
        ax.set_title("Affected Head - Original")

        # Original unaffected mask
        ax = axes[0, 1]
        ax.imshow(self.data['transformed_unaff_mask'], cmap='gray')
        (x1, y1), (x2, y2) = self.data['trans_unaff_major_axis']
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        ax.scatter([x1, x2], [y1, y2], c=['red', 'blue'], s=50)
        ax.scatter(unaff_center[0], unaff_center[1], c='yellow', s=100, marker='*')
        ax.set_title("Unaffected Head - Original")

        # Rotated affected mask
        ax = axes[1, 0]
        ax.imshow(self.rotated_aff_mask, cmap='gray')
        # Draw horizontal line at rotation center
        ax.axhline(aff_center[1], color='cyan', linestyle='--', alpha=0.5)
        ax.scatter(aff_center[0], aff_center[1], c='yellow', s=100, marker='*')
        ax.set_title(f"Affected Head - Rotated ({angle:.1f}°)")

        # Rotated unaffected mask
        ax = axes[1, 1]
        ax.imshow(self.rotated_unaff_mask, cmap='gray')
        ax.axhline(unaff_center[1], color='cyan', linestyle='--', alpha=0.5)
        ax.scatter(unaff_center[0], unaff_center[1], c='yellow', s=100, marker='*')

        # Add COM points
        ax.scatter(com_rotated[0], com_rotated[1], c='green', s=100, marker='.')
        ax.scatter(com_rotated_180[0], com_rotated_180[1], c='purple', s=100, marker='.')
        # ax.scatter(com_orig[0], com_orig[1], c='blue', s=100, marker='.') # troubleshooting com abnormalities


        title = f"Unaffected Head - Rotated ({angle:.1f}°)"
        if self.unaff_flipped:
            title += " (FLIPPED)"
        ax.set_title(title)

        debug_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_axis_rotation.png"
        plt.tight_layout()
        plt.savefig(debug_path, bbox_inches='tight')
        plt.close(fig)

        return debug_path

    def _calculate_pillar_heights(self, mask, laterality):
        """Measure max and average heights for lateral, middle, medial pillars."""
        # Get bounding box of non-zero region
        coords = np.argwhere(mask)
        if coords.size == 0:
            return [0, 0, 0, 0, 0, 0]  # Handle empty mask

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x

        # Define pillar boundaries
        thirds = [min_x, min_x + width / 3, min_x + 2 * width / 3, max_x]

        # Determine pillar order based on laterality
        if laterality == 'R':
            pillars = [('lateral', thirds[0], thirds[1]),
                       ('middle', thirds[1], thirds[2]),
                       ('medial', thirds[2], thirds[3])]
        else:  # 'L'
            pillars = [('medial', thirds[0], thirds[1]),
                       ('middle', thirds[1], thirds[2]),
                       ('lateral', thirds[2], thirds[3])]

        # Measure heights for each pillar
        results = []
        for name, start, end in pillars:
            pillar_mask = mask[:, int(start):int(end) + 1]
            heights = []
            for col in range(pillar_mask.shape[1]):
                col_mask = pillar_mask[:, col]
                if np.any(col_mask):
                    y_indices = np.where(col_mask)[0]
                    height = y_indices.max() - y_indices.min() + 1
                    heights.append(height)

            max_height = np.max(heights) if heights else 0
            avg_height = np.mean(heights) if heights else 0
            results.extend([max_height, avg_height])

        return results

    def measure_pillars(self):
        """Calculate pillar heights for both masks and ratios."""
        # Measure heights for affected and unaffected masks
        aff_heights = self._calculate_pillar_heights(
            self.rotated_aff_mask, self.data['affected_laterality']
        )
        unaff_heights = self._calculate_pillar_heights(
            self.rotated_unaff_mask, self.data['unaffected_laterality']
        )

        # Calculate ratios (avoid division by zero)
        ratios = []
        for aff_val, unaff_val in zip(aff_heights, unaff_heights):
            ratio = aff_val / unaff_val if unaff_val != 0 else float('nan')
            ratios.append(ratio)

        # Store measurements
        pillar_types = ['lateral_max', 'lateral_avg',
                        'middle_max', 'middle_avg',
                        'medial_max', 'medial_avg']

        self.pillar_measurements = {
            'aff_' + k: v for k, v in zip(pillar_types, aff_heights)
        }
        self.pillar_measurements.update({
            'unaff_' + k: v for k, v in zip(pillar_types, unaff_heights)
        })
        self.pillar_measurements.update({
            'ratio_' + k: v for k, v in zip(pillar_types, ratios)
        })

    def _calculate_epiphyseal_quotient(self, mask):
        """Calculate height/width ratio for femoral head mask."""
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return 0.0, 0.0, 0.0

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        width = max_x - min_x
        height = max_y - min_y

        if width > 0:
            eq = height / width
        else:
            eq = 0.0

        return eq, width, height

    def measure_epiphyseal_quotient(self):
        """Calculate EQ (height/width) for both femoral heads."""
        # Calculate EQ for affected head
        aff_eq, aff_width, aff_height = self._calculate_epiphyseal_quotient(self.rotated_aff_mask)
        unaff_eq, unaff_width, unaff_height = self._calculate_epiphyseal_quotient(self.rotated_unaff_mask)

        # Calculate ratio
        eq_ratio = aff_eq / unaff_eq if unaff_eq != 0 else float('nan')

        # Store measurements
        self.eq_measurements = {
            'aff_eq': aff_eq,
            'aff_width': aff_width,
            'aff_height': aff_height,
            'unaff_eq': unaff_eq,
            'unaff_width': unaff_width,
            'unaff_height': unaff_height,
            'eq_ratio': eq_ratio
        }

    @staticmethod
    def find_landmark(mask, laterality):
        """Finds the landmark point (lowest and most lateral) in a mask."""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return (0, 0)
        max_y = np.max(y_indices)  # Lowest point (highest y-value)
        x_at_max_y = x_indices[y_indices == max_y]
        if laterality == 'R':
            x_landmark = np.max(x_at_max_y)  # Rightmost
        else:  # 'L'
            x_landmark = np.min(x_at_max_y)  # Leftmost
        return (int(x_landmark), int(max_y))

    @staticmethod
    def compute_boundaries(mask):
        """Computes top/bottom profiles (per column) and left/right profiles (per row)."""
        H, W = mask.shape
        top = np.full(W, -1, dtype=int)  # -1 indicates no data
        bottom = np.full(W, -1, dtype=int)
        left = np.full(H, -1, dtype=int)
        right = np.full(H, -1, dtype=int)

        for x in range(W):
            col = mask[:, x]
            if np.any(col):
                y_vals = np.where(col)[0]
                top[x] = np.min(y_vals)
                bottom[x] = np.max(y_vals)

        for y in range(H):
            row = mask[y, :]
            if np.any(row):
                x_vals = np.where(row)[0]
                left[y] = np.min(x_vals)
                right[y] = np.max(x_vals)

        return top, bottom, left, right


    def calculate_deformity_index(self):
        # Get masks and laterality
        affected_mask = self.data['affected_mask']
        transformed_unaff_mask = self.data['transformed_unaff_mask']
        affected_laterality = self.data['affected_laterality']
        unaff_width = self.eq_measurements['unaff_width']

        # Find landmarks - CORRECTED CALL (only 2 arguments now)
        aff_landmark = self.find_landmark(affected_mask, affected_laterality)
        unaff_landmark = self.find_landmark(transformed_unaff_mask, affected_laterality)


        # Calculate integer shift
        dx = int(round(aff_landmark[0] - unaff_landmark[0]))
        dy = int(round(aff_landmark[1] - unaff_landmark[1]))
        H, W = affected_mask.shape

        # Create padded canvas to accommodate shifts
        min_x = min(0, dx)
        max_x = max(W - 1, dx + W - 1)
        min_y = min(0, dy)
        max_y = max(H - 1, dy + H - 1)
        new_width = max_x - min_x + 1
        new_height = max_y - min_y + 1

        aff_padded = np.zeros((new_height, new_width), dtype=bool)
        unaff_padded = np.zeros((new_height, new_width), dtype=bool)

        # Place masks in padded canvas
        aff_padded[-min_y: -min_y + H, -min_x: -min_x + W] = affected_mask
        unaff_padded[dy - min_y: dy - min_y + H, dx - min_x: dx - min_x + W] = transformed_unaff_mask

        # Compute boundary profiles
        top_aff, bottom_aff, left_aff, right_aff = self.compute_boundaries(aff_padded)
        top_unaff, bottom_unaff, left_unaff, right_unaff = self.compute_boundaries(unaff_padded)

        # Calculate max height difference (ΔH)
        max_diff_top = 0
        max_diff_bottom = 0
        for x in range(new_width):
            if top_aff[x] != -1 and top_unaff[x] != -1:
                diff = abs(top_aff[x] - top_unaff[x])
                max_diff_top = max(max_diff_top, diff)
            if bottom_aff[x] != -1 and bottom_unaff[x] != -1:
                diff = abs(bottom_aff[x] - bottom_unaff[x])
                max_diff_bottom = max(max_diff_bottom, diff)
        deltaH = max(max_diff_top, max_diff_bottom)

        # Calculate max width difference (ΔW)
        max_diff_left = 0
        max_diff_right = 0
        for y in range(new_height):
            if left_aff[y] != -1 and left_unaff[y] != -1:
                diff = abs(left_aff[y] - left_unaff[y])
                max_diff_left = max(max_diff_left, diff)
            if right_aff[y] != -1 and right_unaff[y] != -1:
                diff = abs(right_aff[y] - right_unaff[y])
                max_diff_right = max(max_diff_right, diff)
        deltaW = max(max_diff_left, max_diff_right)

        # Calculate deformity index
        deformity_index = (deltaH + deltaW) / unaff_width
        self.di_measurements = {
            'deformity_index': deformity_index,
            'deltaH': deltaH,
            'deltaW': deltaW,
            'unaff_diameter': unaff_width,
        }
        return deformity_index, deltaH, deltaW

    def visualize(self):
        """Visualize rotated masks with two subplots:
        Left - EQ & Pillars, Right - Deformity Index"""
        if self.rotated_aff_mask is None:
            raise RuntimeError("Run align_major_axis() first")

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"Patient {self.data['patient_id']} - {self.data['timepoint']}", fontsize=16)

        # --- Subplot 1: EQ & Pillar Visualization ---
        overlay_eq = np.zeros((*self.rotated_aff_mask.shape, 3), dtype=np.uint8)

        # Find contours for mask outlines
        aff_contours, _ = cv2.findContours(
            self.rotated_aff_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        unaff_contours, _ = cv2.findContours(
            self.rotated_unaff_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contour outlines
        cv2.drawContours(overlay_eq, aff_contours, -1, (255, 0, 0), 1)  # Red = affected
        cv2.drawContours(overlay_eq, unaff_contours, -1, (0, 255, 0), 1)  # Green = unaffected

        # Get bounding boxes
        def get_bbox(mask):
            coords = np.argwhere(mask)
            if len(coords) == 0:
                return (0, 0, 0, 0)
            x_min, y_min = coords.min(axis=0)[::-1]
            x_max, y_max = coords.max(axis=0)[::-1]
            return (x_min, y_min, x_max, y_max)

        aff_bbox = get_bbox(self.rotated_aff_mask)
        unaff_bbox = get_bbox(self.rotated_unaff_mask)

        # Draw pillar divisions for affected head
        if aff_bbox[2] - aff_bbox[0] > 0:
            width = aff_bbox[2] - aff_bbox[0]
            div1 = aff_bbox[0] + width / 3
            div2 = aff_bbox[0] + 2 * width / 3

            # Draw pillar division lines
            cv2.line(overlay_eq, (int(div1), aff_bbox[1]), (int(div1), aff_bbox[3]),
                     (255, 255, 255), 2)
            cv2.line(overlay_eq, (int(div2), aff_bbox[1]), (int(div2), aff_bbox[3]),
                     (255, 255, 255), 2)

        # Draw EQ bounding boxes
        for bbox, color in zip([aff_bbox, unaff_bbox], [(0, 0, 255), (0, 255, 255)]):
            if bbox[2] - bbox[0] > 0:
                cv2.rectangle(overlay_eq, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

        ax1.imshow(overlay_eq)
        ax1.set_title("Epiphyseal Quotient & Pillar Analysis\n"
                      f"Affected EQ: {self.eq_measurements['aff_eq']:.2f}, "
                      f"Unaffected EQ: {self.eq_measurements['unaff_eq']:.2f}")
        ax1.axis('off')

        # --- Subplot 2: Deformity Index Visualization ---
        if not hasattr(self, 'di_measurements'):
            self.calculate_deformity_index()

        overlay_di = np.zeros((*self.rotated_aff_mask.shape, 3), dtype=np.uint8)
        aff_contours, _ = cv2.findContours(
            self.data['affected_mask'].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        unaff_contours, _ = cv2.findContours(
            self.data['transformed_unaff_mask'].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        # Draw contour outlines
        cv2.drawContours(overlay_di, aff_contours, -1, (255, 0, 0), 1)
        cv2.drawContours(overlay_di, unaff_contours, -1, (0, 255, 0), 1)

        # Get DI measurements
        deltaH = self.di_measurements['deltaH']
        deltaW = self.di_measurements['deltaW']
        deformity_index = self.di_measurements['deformity_index']
        unaff_width = self.di_measurements['unaff_diameter']


        ax2.imshow(overlay_di)
        ax2.set_title("Deformity Index Analysis\n"
                      f"DI: {deformity_index:.2f}\n"
                      f"Height Diff: {deltaH:.1f}px, "
                      f"Width Diff: {deltaW:.1f}px, "
                      f"Unaffected Diameter: {unaff_width:.1f}px")
        ax2.axis('off')

        # Save combined visualization
        plt.tight_layout()
        vis_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_vis.png"
        plt.savefig(vis_path, bbox_inches='tight', dpi=300)
        plt.close()

        return vis_path

    def get_results(self):
        """
        Return all measurements as a dictionary including:
        - Patient identification
        - Alignment metrics
        - Pillar measurements
        - Epiphyseal quotient measurements
        - Deformity index and components
        """
        # Base information
        result = {
            'patient_id': self.data['patient_id'],
            'timepoint': self.data['timepoint'],
            'affected_laterality': self.data['affected_laterality'],
            'unaffected_laterality': self.data['unaffected_laterality'],
        }

        # Add pillar measurements
        result.update({
            'lateral_ratio': self.pillar_measurements.get('ratio_lateral_max', None),
            'middle_ratio': self.pillar_measurements.get('ratio_middle_max', None),
            'medial_ratio': self.pillar_measurements.get('ratio_medial_max', None),
            'aff_lateral_max': self.pillar_measurements.get('aff_lateral_max', None),
            'aff_lateral_avg': self.pillar_measurements.get('aff_lateral_avg', None),
            'aff_middle_max': self.pillar_measurements.get('aff_middle_max', None),
            'aff_middle_avg': self.pillar_measurements.get('aff_middle_avg', None),
            'aff_medial_max': self.pillar_measurements.get('aff_medial_max', None),
            'aff_medial_avg': self.pillar_measurements.get('aff_medial_avg', None),
            'unaff_lateral_max': self.pillar_measurements.get('unaff_lateral_max', None),
            'unaff_lateral_avg': self.pillar_measurements.get('unaff_lateral_avg', None),
            'unaff_middle_max': self.pillar_measurements.get('unaff_middle_max', None),
            'unaff_middle_avg': self.pillar_measurements.get('unaff_middle_avg', None),
            'unaff_medial_max': self.pillar_measurements.get('unaff_medial_max', None),
            'unaff_medial_avg': self.pillar_measurements.get('unaff_medial_avg', None),

        })

        # Add EQ measurements
        result.update({
            'eq_ratio': self.eq_measurements.get('eq_ratio', None),
            'aff_eq': self.eq_measurements.get('aff_eq', None),
            'aff_width': self.eq_measurements.get('aff_width', None),
            'aff_height': self.eq_measurements.get('aff_height', None),
            'unaff_eq': self.eq_measurements.get('unaff_eq', None),
            'unaff_width': self.eq_measurements.get('unaff_width', None),
            'unaff_height': self.eq_measurements.get('unaff_height', None)

        })

        # Add DI measurements
        result.update({
            'deformity_index': self.di_measurements.get('deformity_index', None),
            'deltaH': self.di_measurements.get('deltaH', None),
            'deltaW': self.di_measurements.get('deltaW', None),
            'unaff_diameter': self.di_measurements.get('unaff_diameter', None),
        })

        return result

        # ... (rest of the class remains unchanged)
    def process(self):
        """Complete processing pipeline for a single patient/timepoint."""
        # Perform alignment
        self.align_major_axis()

        # Continue with measurements
        self.measure_pillars()
        self.measure_epiphyseal_quotient()
        self.calculate_deformity_index()

        # Visualize results
        vis_path = self.visualize()

        # Return visualization path and results dictionary
        return vis_path, self.get_results()