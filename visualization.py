import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from PIL import Image
from skimage import measure
from measurement import DIMeasurement


class MeasurementVisualizer:
    def __init__(self, analyzer, output_folder):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def visualize_di(self):
        """Visualize DI measurement with landmark-aligned masks"""
        di_data = self.analyzer.di_measurements
        aff_mask = di_data.get('aff_padded', None)
        unaff_mask = di_data.get('unaff_padded', None)

        if aff_mask is None or unaff_mask is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')  # Preserve proportions

        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        # Add labels
        self._draw_label(ax, "Affected", 'red', 0.05, 0.95)
        self._draw_label(ax, "Unaffected", 'lime', 0.05, 0.90)

        # Add landmark markers
        aff_landmark = DIMeasurement.find_landmark(aff_mask, self.data['affected_laterality'])
        unaff_landmark = DIMeasurement.find_landmark(unaff_mask, self.data['affected_laterality'])
        ax.scatter(aff_landmark[0], aff_landmark[1], s=80, c='red', marker='o', edgecolor='white')
        ax.scatter(unaff_landmark[0], unaff_landmark[1], s=80, c='lime', marker='o', edgecolor='white')

        # Add title with measurements
        title = (f"Patient {self.data['patient_id']} - {self.data['timepoint']}\n"
                 f"DI: {di_data.get('deformity_index', 'N/A'):.2f}, "
                 f"ΔH: {di_data.get('deltaH', 'N/A'):.1f}, "
                 f"ΔW: {di_data.get('deltaW', 'N/A'):.1f}, "
                 f"Unaff Diam: {di_data.get('unaff_diameter', 'N/A'):.1f}")
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Save and close
        di_folder = self.output_folder / "di_viz"
        di_folder.mkdir(exist_ok=True)
        save_path = di_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_di.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path

    # Update the visualize_lateral_pillar method
    def visualize_lateral_pillar(self):
        """Visualize lateral pillars with thirds division and COM markers"""
        aff_mask = self.analyzer.rotated_aff_mask
        unaff_mask = self.analyzer.rotated_unaff_mask
        pillar_data = self.analyzer.pillar_measurements

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')  # Preserve proportions

        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        # Add labels
        self._draw_label(ax, "Affected", 'red', 0.05, 0.95)
        self._draw_label(ax, "Unaffected", 'lime', 0.05, 0.90)

        # Draw pillar divisions
        self._draw_pillar_divisions(ax, aff_mask, self.data['affected_laterality'], 'red')
        self._draw_pillar_divisions(ax, unaff_mask, self.data['affected_laterality'], 'lime')

        # Calculate and plot COM
        aff_com = self.analyzer.rotated_aff_com
        unaff_com = self.analyzer.rotated_unaff_com

        # Plot COM markers
        ax.scatter(aff_com[0], aff_com[1], s=100, c='red', marker='x', linewidth=2)
        ax.scatter(unaff_com[0], unaff_com[1], s=100, c='lime', marker='x', linewidth=2)

        # Add COM labels
        ax.text(aff_com[0], aff_com[1] + 10, 'Aff COM',
                color='red', ha='center', va='bottom', fontsize=10)
        ax.text(unaff_com[0], unaff_com[1] + 10, 'Unaff COM',
                color='lime', ha='center', va='bottom', fontsize=10)

        # Add title with measurements
        title = (f"Patient {self.data['patient_id']} - {self.data['timepoint']}\n"
                 f"Lat Ratio: {pillar_data.get('ratio_lateral_avg', 'N/A'):.2f}, "
                 f"Med Ratio: {pillar_data.get('ratio_medial_avg', 'N/A'):.2f}, "
                 f"Herring: {pillar_data.get('herring_class', 'N/A')}")
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Save and close
        lateral_folder = self.output_folder / "lateral_pillar_viz"
        lateral_folder.mkdir(exist_ok=True)
        save_path = lateral_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_pillar.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path


    def visualize_eq(self):
        """Visualize EQ with bounding boxes"""
        aff_mask = self.analyzer.rotated_aff_mask
        unaff_mask = self.analyzer.rotated_unaff_mask
        eq_data = self.analyzer.eq_measurements

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')  # Preserve proportions

        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        # Add labels
        self._draw_label(ax, "Affected", 'red', 0.05, 0.95)
        self._draw_label(ax, "Unaffected", 'lime', 0.05, 0.90)

        # Draw bounding boxes
        self._draw_bounding_box(ax, aff_mask, 'red')
        self._draw_bounding_box(ax, unaff_mask, 'lime')

        # Add dimensions
        self._draw_dimensions(ax, aff_mask, 'red')
        self._draw_dimensions(ax, unaff_mask, 'lime')

        # Add title with measurements
        title = (f"Patient {self.data['patient_id']} - {self.data['timepoint']}\n"
                 f"EQ Ratio: {eq_data.get('eq_ratio', 'N/A'):.2f}, "
                 f"Aff EQ: {eq_data.get('aff_eq', 'N/A'):.2f}, "
                 f"Unaff EQ: {eq_data.get('unaff_eq', 'N/A'):.2f}")
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Save and close
        eq_folder = self.output_folder / "eq_viz"
        eq_folder.mkdir(exist_ok=True)
        save_path = eq_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_eq.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path

    def _draw_mask_outline(self, ax, mask, color):
        """Draw outline of a mask"""
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    def _draw_pillar_divisions(self, ax, mask, laterality, color):
        """Draw vertical lines dividing pillars with labels"""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x

        # Calculate pillar boundaries
        div1 = min_x + 0.25 * width
        div2 = min_x + 0.75 * width

        # Draw vertical lines
        ax.axvline(x=div1, color=color, linestyle='--', alpha=0.7)
        ax.axvline(x=div2, color=color, linestyle='--', alpha=0.7)

        # Label pillars
        mid_y = (min_y + max_y) / 2
        if laterality == 'R':
            # Right hip: lateral | middle | medial
            ax.text(div1 - 0.05 * width, mid_y, "Lateral", ha='right', va='center', color=color)
            ax.text((div1 + div2) / 2, mid_y, "Middle", ha='center', va='center', color=color)
            ax.text(div2 + 0.05 * width, mid_y, "Medial", ha='left', va='center', color=color)
        else:
            # Left hip: medial | middle | lateral
            ax.text(div1 - 0.05 * width, mid_y, "Medial", ha='right', va='center', color=color)
            ax.text((div1 + div2) / 2, mid_y, "Middle", ha='center', va='center', color=color)
            ax.text(div2 + 0.05 * width, mid_y, "Lateral", ha='left', va='center', color=color)

    def _draw_bounding_box(self, ax, mask, color):
        """Draw bounding box around mask"""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        # Draw rectangle
        rect = plt.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle='--'
        )
        ax.add_patch(rect)

    def _draw_dimensions(self, ax, mask, color):
        """Draw width and height dimensions"""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        # Draw width dimension
        mid_y = min_y - height * 0.05
        ax.plot([min_x, max_x], [mid_y, mid_y], color='gray', linewidth=1)
        ax.text((min_x + max_x) / 2, mid_y, f"W: {width:.1f}",
                ha='center', va='bottom', color='gray')

        # Draw height dimension
        mid_x = max_x + width * 0.05
        ax.plot([mid_x, mid_x], [min_y, max_y], color='gray', linewidth=1)
        ax.text(mid_x, (min_y + max_y) / 2, f"H: {height:.1f}",
                ha='left', va='center', color='gray', rotation=90)

    def _draw_label(self, ax, text, color, x, y):
        """Draw a text label at normalized coordinates"""
        ax.text(x, y, text,
                transform=ax.transAxes,
                color=color,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


class FinalMeasurementVisualizer:
    def __init__(self, analyzer, img_folder, output_folder):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.output_folder = Path(output_folder) / "final viz"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.img_folder = Path(img_folder)
        self.original_image = self._load_original_image()
        # Load unaffected image
        self.unaffected_image = self._load_unaffected_image()

    def _load_original_image(self):
        img_path = self.img_folder / self.data['aff_img_info']['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Original image not found: {img_path}")
        return np.array(Image.open(img_path).convert('L'))

    def _load_unaffected_image(self):
        img_path = self.img_folder / self.data['unaff_img_info']['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Unaffected image not found: {img_path}")
        return np.array(Image.open(img_path).convert('L'))

    def _draw_mask_outline(self, ax, mask, color):
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    def _plot_affected_original(self, ax):
        ax.imshow(self.original_image, cmap='gray')
        ax.set_title("Affected Femoral Head (Original)")
        ax.axis('off')

    def _plot_unaffected_original(self, ax):
        ax.imshow(self.unaffected_image, cmap='gray')
        ax.set_title("Unaffected Femoral Head (Original)")
        ax.axis('off')

    def _plot_lateral_pillar(self, ax):
        ax.imshow(self.original_image, cmap='gray')
        self._draw_mask_outline(ax, self.data['affected_mask'], color='red')

        # Get the affected mask
        aff_mask = self.data['affected_mask']
        coords = np.argwhere(aff_mask)
        if coords.size == 0:
            # Skip if no mask
            pass
        else:
            # Get mask bounding box
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            # Get major axis endpoints
            major_axis = self.data['aff_major_axis']
            if major_axis:
                # Extract endpoints
                (x1, y1), (x2, y2) = major_axis

                # Calculate vector along major axis
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)

                if length > 0:  # Ensure valid axis
                    # Calculate unit vector along major axis
                    ux = dx / length
                    uy = dy / length

                    # Calculate unit vector perpendicular to major axis
                    vx = -uy
                    vy = ux

                    # Calculate quarter points along major axis
                    quarter1 = (x1 + 0.25 * dx, y1 + 0.25 * dy)
                    quarter3 = (x1 + 0.75 * dx, y1 + 0.75 * dy)

                    # Draw lines at quarter points perpendicular to major axis
                    for point in [quarter1, quarter3]:
                        px, py = point
                        # Calculate line endpoints at top and bottom of mask
                        top_x = px - vx * abs(max_x-min_x) * 1/2
                        top_y = py - vy * abs(max_y-min_y) * 1/2
                        bottom_x = px + vx * abs(max_x-min_x) * 1/2
                        bottom_y = py + vy * abs(max_y-min_y) * 1/2

                        # Draw the line from top to bottom of mask
                        ax.plot([top_x, bottom_x], [top_y, bottom_y],
                                color='blue', linewidth=2, linestyle='-')

        ratio = self.analyzer.pillar_measurements.get('ratio_lateral_avg', None)
        ratio_txt = f"{ratio:.2f}" if ratio is not None else "N/A"
        classification = self.analyzer.pillar_measurements.get('herring_class', None)
        classification_txt = classification if classification is not None else "N/A"

        ax.set_title(f"Lateral Pillar Comparison\nLateral Pillar Ratio: {ratio_txt} ({classification_txt})")
        ax.axis('off')

    def _plot_eq(self, ax):  # Modified: removed height/width bars
        ax.imshow(self.original_image, cmap='gray')
        self._draw_mask_outline(ax, self.data['affected_mask'], color='red')

        unaffeq = self.analyzer.eq_measurements.get('unaff_eq', None)
        unaffeq_txt = f"{unaffeq:.2f}" if unaffeq is not None else "N/A"
        affeq = self.analyzer.eq_measurements.get('aff_eq', None)
        affeq_txt = f"{affeq:.2f}" if affeq is not None else "N/A"
        eq_ratio = self.analyzer.eq_measurements.get('eq_ratio', None)
        eq_ratio_txt = f"{eq_ratio:.2f}" if affeq is not None else "N/A"

        ax.set_title(f"Epiphyseal Quotient\nUnaffected EI: {unaffeq_txt}\nAffected EI: {affeq_txt}\nEQ: {eq_ratio_txt}")

        ax.axis('off')

    def _plot_deformity(self, ax):
        di_data = self.analyzer.di_measurements
        aff_mask = di_data.get('aff_padded', None)
        unaff_mask = di_data.get('unaff_padded', None)


        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')
        ax.imshow(self.original_image, cmap='gray')

        di = self.analyzer.di_measurements.get('deformity_index', None)
        di_txt = f"{di:.2f}" if di is not None else "N/A"
        ax.set_title(f"Deformity Visualization\nDeformity Index: {di_txt}")
        ax.axis('off')

    def visualize_quad(self):
        # Create 2x3 grid (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Patient {self.data['patient_id']} - {self.data['timepoint']}", fontsize=16)

        # Top row: original images (first two columns)
        self._plot_affected_original(axes[0, 0])
        self._plot_unaffected_original(axes[0, 1])

        # Hide the top-right cell
        axes[0, 2].axis('off')

        # Bottom row: all three measurements
        self._plot_lateral_pillar(axes[1, 0])
        self._plot_eq(axes[1, 1])
        self._plot_deformity(axes[1, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save to final viz folder
        quad_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_quad_vis.png"
        plt.savefig(quad_path, bbox_inches='tight', dpi=300)  # Increased DPI
        plt.close(fig)

        return quad_path