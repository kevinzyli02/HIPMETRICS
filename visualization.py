import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from PIL import Image
from skimage import measure


class FinalMeasurementVisualizer:
    def __init__(self, analyzer, img_folder, output_folder):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.output_folder = Path(output_folder)
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
        self._draw_mask_outline(ax, self.data['transformed_unaff_mask'], color='lime')

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
        ax.imshow(self.original_image, cmap='gray')
        self._draw_mask_outline(ax, self.data['affected_mask'], color='red')
        self._draw_mask_outline(ax, self.data['transformed_unaff_mask'], color='lime')

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
        quad_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_quad_vis.png"
        plt.savefig(quad_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return quad_path