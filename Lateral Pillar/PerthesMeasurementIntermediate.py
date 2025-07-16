import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import math


class PerthesMeasurements:
    def __init__(self, aligned_results, output_dir):
        """
        Initialize with aligned femoral head results.

        Args:
            aligned_results (list): List of dictionaries from femoral head alignment
            output_dir (str): Directory to save reports
        """
        self.aligned_results = aligned_results
        self.output_dir = output_dir
        self.measurements = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_rotation_angle(self, major_axis_endpoints):
        """
        Calculate rotation angle to make major axis horizontal.

        Args:
            major_axis_endpoints (tuple): ((x1,y1),(x2,y2)) of major axis

        Returns:
            float: Rotation angle in degrees
        """
        p1, p2 = major_axis_endpoints
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Calculate angle in radians
        angle_rad = math.atan2(dy, dx)

        # Convert to degrees and ensure horizontal orientation
        angle_deg = math.degrees(angle_rad)

        # Normalize to horizontal orientation (0° or 180°)
        if abs(angle_deg) > 45 and abs(angle_deg) < 135:
            angle_deg += 90

        return -angle_deg  # Negative for clockwise rotation

    def _rotate_mask(self, mask, angle_deg):
        """
        Rotate mask around its center of mass.

        Args:
            mask (ndarray): Binary mask
            angle_deg (float): Rotation angle in degrees

        Returns:
            rotated_mask: Rotated binary mask
        """
        # Find center of mass
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return mask

        center_yx = np.mean(coords, axis=0)
        center_xy = (center_yx[1], center_yx[0])  # Convert to (x, y)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            center_xy,
            angle_deg,
            1.0  # scale
        )

        # Apply rotation
        rotated = cv2.warpAffine(
            mask.astype(np.uint8) * 255,
            rotation_matrix,
            (mask.shape[1], mask.shape[0]),  # (width, height)
            flags=cv2.INTER_NEAREST
        )

        return rotated > 127

    def _get_lateral_pillar_height(self, rotated_mask, side):
        """
        Calculate lateral pillar height from rotated mask.

        Args:
            rotated_mask (ndarray): Mask with major axis horizontal
            side (str): 'left' or 'right'

        Returns:
            float: Maximum height in lateral third
        """
        # Get bounding box of non-zero regions
        coords = np.argwhere(rotated_mask)
        if len(coords) == 0:
            return 0.0

        min_y, min_x = np.min(coords, axis=0)
        max_y, max_x = np.max(coords, axis=0)

        width = max_x - min_x
        lateral_third = width // 3

        if side == 'right':
            x_start = max_x - lateral_third
            x_end = max_x
        else:  # left
            x_start = min_x
            x_end = min_x + lateral_third

        # Extract lateral third region
        strip = rotated_mask[min_y:max_y + 1, x_start:x_end]
        if not np.any(strip):
            return 0.0

        # Find top and bottom edges
        y_coords = np.where(strip.any(axis=1))[0]
        if len(y_coords) == 0:
            return 0.0

        return y_coords[-1] - y_coords[0] + 1

    def _get_epiphyseal_quotient(self, rotated_mask):
        """
        Calculate EQ from rotated mask.

        Args:
            rotated_mask (ndarray): Mask with major axis horizontal

        Returns:
            tuple: (EQ, height, width)
        """
        points = np.argwhere(rotated_mask)
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        min_y, min_x = np.min(points, axis=0)
        max_y, max_x = np.max(points, axis=0)

        width = max_x - min_x
        height = max_y - min_y

        if width > 0:
            eq = height / width
        else:
            eq = 0.0

        return eq, height, width

    def calculate_all_measurements(self):
        """
        Calculate all Perthes measurements for all aligned results.
        """
        self.measurements = []

        for result in self.aligned_results:
            # Get rotation angle from affected head
            angle = self._get_rotation_angle(result['aff_major_axis'])

            # Rotate both masks using the same angle
            aff_rotated = self._rotate_mask(result['affected_mask'], angle)
            unaff_rotated = self._rotate_mask(result['transformed_unaff_mask'], angle)

            # Get sides
            aff_side = result['affected_laterality'].lower()
            unaff_side = result['unaffected_laterality'].lower()

            # Process affected head
            aff_lp_height = self._get_lateral_pillar_height(aff_rotated, aff_side)
            aff_eq, aff_height, aff_width = self._get_epiphyseal_quotient(aff_rotated)

            # Process unaffected head
            unaff_lp_height = self._get_lateral_pillar_height(unaff_rotated, unaff_side)
            unaff_eq, unaff_height, unaff_width = self._get_epiphyseal_quotient(unaff_rotated)

            # Calculate ratios
            if unaff_lp_height > 0:
                lp_ratio = aff_lp_height / unaff_lp_height
            else:
                lp_ratio = float('nan')

            if unaff_eq > 0:
                di = 1 - (aff_eq / unaff_eq)
            else:
                di = float('nan')

            # Store measurements
            self.measurements.append({
                'patient_id': result['patient_id'],
                'timepoint': result['timepoint'],
                'affected_lateral': aff_lp_height,
                'unaffected_lateral': unaff_lp_height,
                'LP_ratio': lp_ratio,
                'affected_EQ': aff_eq,
                'unaffected_EQ': unaff_eq,
                'affected_height': aff_height,
                'affected_width': aff_width,
                'unaffected_height': unaff_height,
                'unaffected_width': unaff_width,
                'deformity_index': di,
                'com_distance': result['dist'],
                'affected_side': aff_side,
                'unaffected_side': unaff_side,
                'rotated_affected': aff_rotated,
                'rotated_unaffected': unaff_rotated,
                'result_data': result
            })

    def _create_visualization(self, measurement):
        """
        Create visualization for one measurement case.

        Returns:
            fig: Matplotlib figure
        """
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3)

        # Panel 1: Unaffected side
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_head(ax1,
                        measurement['rotated_unaffected'],
                        f"Unaffected {measurement['unaffected_side'].capitalize()} Head",
                        'blue',
                        measurement['unaffected_side'])

        # Panel 2: Affected side with overlay
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_head(ax2,
                        measurement['rotated_affected'],
                        f"Affected {measurement['affected_side'].capitalize()} Head",
                        'red',
                        measurement['affected_side'])
        # Overlay unaffected in blue with transparency
        self._plot_head(ax2,
                        measurement['rotated_unaffected'],
                        None,
                        'blue',
                        measurement['unaffected_side'],
                        alpha=0.3)

        # Panel 3: Measurements
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_measurements(ax3, measurement)

        plt.tight_layout()
        return fig

    def _plot_head(self, ax, mask, title, color, side, alpha=1.0):
        """Plot a single head mask without flipping."""
        # Create RGB image
        rgb = np.zeros((*mask.shape, 3))
        if color == 'red':
            rgb[mask] = [1, 0, 0]
        else:  # blue
            rgb[mask] = [0, 0, 1]

        # Display without flipping
        ax.imshow(rgb, alpha=alpha, origin='upper')
        if title:
            ax.set_title(title)
        ax.axis('off')

        # Add lateral third indicator
        width = mask.shape[1]
        lateral_third = width // 3
        if side == 'right':
            ax.axvline(width - lateral_third, color='yellow', linestyle='--')
        else:
            ax.axvline(lateral_third, color='yellow', linestyle='--')

        # Add height indicator
        coords = np.argwhere(mask)
        if len(coords) > 0:
            min_y, min_x = np.min(coords, axis=0)
            max_y, max_x = np.max(coords, axis=0)
            height = max_y - min_y + 1
            width = max_x - min_x + 1

            # Place width at bottom
            ax.text(min_x + width // 2, max_y + 5, f"W: {width:.1f}",
                    color='white', fontsize=10, ha='center',
                    bbox=dict(facecolor='black', alpha=0.5))

            # Place height on right side
            ax.text(max_x + 5, min_y + height // 2, f"H: {height:.1f}",
                    color='white', fontsize=10, va='center', rotation=270,
                    bbox=dict(facecolor='black', alpha=0.5))

    def _plot_measurements(self, ax, measurement):
        """Plot measurement summary."""
        ax.axis('off')
        ax.set_title("Measurement Summary", fontsize=16)

        # Create text content
        text_content = [
            f"Patient: {measurement['patient_id']}",
            f"Timepoint: {measurement['timepoint']}",
            "",
            "Lateral Pillar:",
            f"Affected height: {measurement['affected_lateral']:.1f} px",
            f"Unaffected height: {measurement['unaffected_lateral']:.1f} px",
            f"LP Ratio: {measurement['LP_ratio']:.2f}",
            "",
            "Epiphyseal Quotient:",
            f"Affected EQ: {measurement['affected_EQ']:.2f}",
            f"Unaffected EQ: {measurement['unaffected_EQ']:.2f}",
            f"Deformity Index: {measurement['deformity_index']:.2f}",
            "",
            f"COM Distance: {measurement['com_distance']:.1f} px"
        ]

        # Add text to axis
        ax.text(0.5, 0.5, "\n".join(text_content),
                fontsize=14,
                ha='center', va='center',
                bbox=dict(facecolor='lightgray', alpha=0.5))

    def generate_reports(self):
        """Generate all reports and save Excel file."""
        self.calculate_all_measurements()

        # Save Excel file
        excel_path = os.path.join(self.output_dir, 'perthes_measurements.xlsx')
        df = pd.DataFrame(self.measurements)

        # Drop image data before saving to Excel
        cols_to_drop = ['rotated_affected', 'rotated_unaffected', 'result_data']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        df.to_excel(excel_path, index=False)

        # Generate visual reports
        for measurement in self.measurements:
            fig = self._create_visualization(measurement)
            report_path = os.path.join(
                self.output_dir,
                f"patient_{measurement['patient_id']}_{measurement['timepoint']}_report.png"
            )
            fig.savefig(report_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

        print(f"Generated {len(self.measurements)} reports in {self.output_dir}")