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

    def _compute_rotation_angle(self, axis_endpoints):
        """Calculate rotation angle to make major axis horizontal."""
        (x1, y1), (x2, y2) = axis_endpoints
        dx, dy = x2 - x1, y2 - y1
        return -np.degrees(np.arctan2(dy, dx))  # Negative to counter-clockwise rotation

    def _rotate_mask(self, mask, angle):
        """Rotate mask around its center of mass."""
        # Calculate center of mass
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return mask  # Return original if empty mask
        cx, cy = np.mean(x_indices), np.mean(y_indices)
        # Rotate mask around COM
        rotated = rotate(mask.astype(float), angle, center=(cx, cy),
                         preserve_range=True, mode='constant', cval=0)
        return rotated > 0.5  # Re-binarize mask

    def align_major_axis(self):
        """Rotate masks to make affected major axis horizontal."""
        # Calculate rotation angle from affected major axis
        angle = self._compute_rotation_angle(self.data['aff_major_axis'])

        # Rotate both masks
        self.rotated_aff_mask = self._rotate_mask(
            self.data['affected_mask'], angle
        )
        self.rotated_unaff_mask = self._rotate_mask(
            self.data['transformed_unaff_mask'], angle
        )

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

    def visualize(self):
        """Visualize rotated masks with pillar divisions and axes."""
        if self.rotated_aff_mask is None:
            raise RuntimeError("Run align_major_axis() first")

        # Create RGB overlay
        overlay = np.zeros((*self.rotated_aff_mask.shape, 3), dtype=np.uint8)
        overlay[self.rotated_aff_mask] = [255, 0, 0]  # Red for affected
        overlay[self.rotated_unaff_mask] = [0, 255, 0]  # Green for unaffected

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
        if aff_bbox[2] - aff_bbox[0] > 0:  # Check if valid bbox
            width = aff_bbox[2] - aff_bbox[0]
            div1 = aff_bbox[0] + width / 3
            div2 = aff_bbox[0] + 2 * width / 3

            # Draw lines
            cv2.line(overlay, (int(div1), 0), (int(div1), overlay.shape[0]),
                     (255, 255, 255), 2)
            cv2.line(overlay, (int(div2), 0), (int(div2), overlay.shape[0]),
                     (255, 255, 255), 2)

        # Draw bounding boxes and major axes
        for bbox, color in zip([aff_bbox, unaff_bbox], [(0, 0, 255), (0, 255, 255)]):
            if bbox[2] - bbox[0] > 0:  # Only draw if valid
                # Draw bounding box
                cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

                # Draw major axis (horizontal)
                y_center = (bbox[1] + bbox[3]) // 2
                cv2.line(overlay, (bbox[0], y_center), (bbox[2], y_center), color, 2)

        # Create and save visualization
        plt.figure(figsize=(10, 8))
        plt.title(f"Patient {self.data['patient_id']} - {self.data['timepoint']}")
        plt.imshow(overlay)
        plt.axis('off')

        # Save to output folder
        vis_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_vis.png"
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()
        return vis_path

    def save_to_excel(self):
        """Save all measurements to Excel file in output folder."""
        if not self.pillar_measurements or not self.eq_measurements:
            raise RuntimeError("Run measure_pillars() and measure_epiphyseal_quotient() first")

        # Create data row
        row = {
            'patient_id': self.data['patient_id'],
            'timepoint': self.data['timepoint'],
            'orientation': self.data['orientation'],
            'dist': self.data['dist'],
            **self.pillar_measurements,
            **self.eq_measurements
        }

        excel_path = self.output_folder / "measurements.xlsx"

        # Create or append to Excel
        if excel_path.exists():
            df = pd.read_excel(excel_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_excel(excel_path, index=False)
        return excel_path

    def process(self):
        """Complete processing pipeline for a single patient/timepoint."""
        self.align_major_axis()
        self.measure_pillars()
        self.measure_epiphyseal_quotient()
        vis_path = self.visualize()
        excel_path = self.save_to_excel()
        return vis_path, excel_path