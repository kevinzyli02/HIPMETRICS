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

        # Calculate rotated COM positions
        com_rotated = self._rotate_point(com_orig, unaff_center, angle)
        com_rotated_180 = self._rotate_point(com_rotated, unaff_center, 180)

        # Compare y-values and rotate 180° if needed
        if com_rotated_180[1] < com_rotated[1]:
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
        return self._visualize_rotation(angle, aff_center, unaff_center, com_rotated,com_rotated_180)

    def _visualize_rotation(self, angle, aff_center, unaff_center, com_rotated,com_rotated_180):
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
        ax.scatter(com_rotated[0], com_rotated[1], c='green', s=100, marker='o')
        ax.scatter(com_rotated_180[0], com_rotated_180[1], c='purple', s=100, marker='x')

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

    def calculate_deformity_index(self, visualize=False):
        """
        Calculate the deformity index and store it in self.deformity_index.

        Parameters:
        visualize (bool): Whether to visualize the deformity index calculation (default is False).

        Returns:
        float: Deformity index.
        """
        affected_mask = self.data['affected_mask']
        transformed_unaff_mask = self.data['transformed_unaff_mask']
        laterality = self.data['affected_laterality']
        unaff_width = self.eq_measurements['unaff_width']  # diameter of unaffected hip

        # Step 1: Find the lowest points for alignment
        # Get the lowest y-point of the affected mask
        affected_y_min = np.min(np.where(affected_mask == 1)[0])  # Y position of lowest point in affected mask
        affected_x_min = np.min(np.where(affected_mask == 1)[1])  # X position of lowest point in affected mask

        # Get the lowest y-point of the transformed unaffected mask (aligned based on laterality)
        if laterality == 'R':  # Right side
            unaff_y_min = np.min(np.where(transformed_unaff_mask == 1)[0])
            unaff_x_min = np.max(np.where(transformed_unaff_mask == 1)[1])  # Rightmost point
        else:  # Left side
            unaff_y_min = np.min(np.where(transformed_unaff_mask == 1)[0])
            unaff_x_min = np.min(np.where(transformed_unaff_mask == 1)[1])  # Leftmost point

        # Step 2: Align the masks by translating the transformed unaffected mask
        y_translation = affected_y_min - unaff_y_min
        x_translation = affected_x_min - unaff_x_min

        # Translate the unaffected mask
        aligned_unaff_mask = np.roll(transformed_unaff_mask, shift=(y_translation, x_translation), axis=(0, 1))

        # Step 3: Calculate bounding boxes for both masks
        def get_bbox(mask):
            coords = np.argwhere(mask)
            if len(coords) == 0:
                return (0, 0, 0, 0)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            return (x_min, y_min, x_max, y_max)

        aff_bbox = get_bbox(affected_mask)
        unaff_bbox = get_bbox(aligned_unaff_mask)

        # Unpack bounding boxes
        aff_xmin, aff_ymin, aff_xmax, aff_ymax = aff_bbox
        unaff_xmin, unaff_ymin, unaff_xmax, unaff_ymax = unaff_bbox

        # Calculate height difference (top position difference)
        height_diff = abs(aff_ymax - unaff_ymax)

        # Calculate width difference based on laterality
        if laterality == 'R':
            # Right hip: difference between affected rightmost and unaffected leftmost
            width_diff = abs(aff_xmax - unaff_xmin)
        else:
            # Left hip: difference between unaffected rightmost and affected rightmost
            width_diff = abs(unaff_xmax - aff_xmax)

        # Step 4: Calculate the deformity index
        deformity_index = (height_diff + width_diff) / unaff_width

        # Store the result
        self.deformity_index = deformity_index

        # If visualize is True, generate the visualization
        if visualize:
            self._visualize_deformity_index(
                affected_mask, aligned_unaff_mask,
                laterality, height_diff, width_diff
            )

        return deformity_index

    def _visualize_deformity_index(self, affected_mask, aligned_unaff_mask, laterality, height_diff, width_diff):
        """
        Visualize and save the deformity index calculation.

        Parameters:
        affected_mask (ndarray): Binary mask of the affected femoral head.
        aligned_unaff_mask (ndarray): Transformed binary mask of the unaffected femoral head.
        laterality (str): Laterality of the affected femoral head ('L' or 'R').
        height_diff (int): Maximum height difference between the aligned masks.
        width_diff (int): Maximum width difference between the aligned masks.
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Display the affected and transformed unaffected masks
        ax.imshow(affected_mask, cmap='Blues', alpha=0.6, label='Affected Mask')
        ax.imshow(aligned_unaff_mask, cmap='Reds', alpha=0.6, label='Transformed Unaffected Mask')

        # Get the coordinates of the lowest points for alignment
        affected_y_min = np.min(np.where(affected_mask == 1)[0])
        affected_x_min = np.min(np.where(affected_mask == 1)[1])

        if laterality == 'R':
            unaff_y_min = np.min(np.where(aligned_unaff_mask == 1)[0])
            unaff_x_min = np.max(np.where(aligned_unaff_mask == 1)[1])  # Rightmost point
        else:  # Left side
            unaff_y_min = np.min(np.where(aligned_unaff_mask == 1)[0])
            unaff_x_min = np.min(np.where(aligned_unaff_mask == 1)[1])  # Leftmost point

        # Mark the lowest points of both masks
        ax.plot(affected_x_min, affected_y_min, 'go', label='Affected Mask Alignment')
        ax.plot(unaff_x_min, unaff_y_min, 'ro', label='Unaffected Mask Alignment')

        # Add arrows for the height and width differences
        ax.arrow(affected_y_min, affected_y_min, 0, height_diff, head_width=10, head_length=5, fc='green', ec='green')
        ax.arrow(affected_x_min, affected_x_min, width_diff, 0, head_width=10, head_length=5, fc='blue', ec='blue')

        # Add text for deformity index
        ax.text(0.5, 0.05, f'Deformity Index: {self.deformity_index:.2f}', ha='center', va='center', fontsize=14,
                color='black', transform=ax.transAxes)

        # Set labels and title
        ax.set_title('Deformity Index Visualization')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Show the legend
        ax.legend(loc='upper right')

        # Save the visualization
        debug_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_deformity_index.png"
        plt.savefig(debug_path, bbox_inches='tight')
        plt.close(fig)

        return debug_path

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

    def save_to_excel(self, filename: str, output_dir: str = None) -> Path:
        """
        Save all measurements to Excel file in specified output directory.
        Appends to existing file or creates new one. Returns path to saved file.

        Args:
            filename: Excel file name (e.g., "results.xlsx")
            output_dir: Directory to save file (default: class output_folder)

        Returns:
            Path to saved Excel file
        """
        if not self.pillar_measurements or not self.eq_measurements:
            raise RuntimeError("Run measure_pillars() and measure_epiphyseal_quotient() first")
        if self.deformity_index is None:
            raise RuntimeError("Run calculate_deformity_index() first")

        # Create data row with deformity index
        row = {
            'patient_id': self.data['patient_id'],
            'timepoint': self.data['timepoint'],
            'orientation': self.data['orientation'],
            'dist': self.data['dist'],
            **self.pillar_measurements,
            **self.eq_measurements,
            'deformity_index': self.deformity_index  # Add deformity index
        }

        # Determine output directory
        target_dir = Path(output_dir) if output_dir else self.output_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create full filepath
        excel_path = target_dir / filename

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
        # Perform alignment
        self.align_major_axis()

        # Continue with measurements
        self.measure_pillars()
        self.measure_epiphyseal_quotient()
        self.calculate_deformity_index(visualize=True)  # Calculate and visualize deformity index

        # Visualize results
        vis_path = self.visualize()
        excel_path = self.save_to_excel("results.xlsx")

        return vis_path, excel_path