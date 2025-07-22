import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


class FemoralHeadVisualizer:
    def __init__(self, analyzer, output_folder):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _plot_affected_only(self, ax):
        """Plot 1: Original radiograph with affected mask overlay"""
        ax.imshow(self.data['original_image'], cmap='gray')
        mask = self.data['affected_mask']
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 0, 1, 0.3]  # Blue with transparency
        ax.imshow(overlay)
        ax.set_title("Affected Femoral Head")
        ax.axis('off')

    def _plot_unaffected_overlay(self, ax):
        """Plot 2: Original radiograph with transformed unaffected mask overlay"""
        ax.imshow(self.data['original_image'], cmap='gray')
        mask = self.data['transformed_unaff_mask']
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 1, 0, 0.3]  # Green with transparency
        ax.imshow(overlay)

        # Add lateral pillar ratio to title
        ratio = self.analyzer.pillar_measurements.get('ratio_lateral_max', None)
        ratio_txt = f"{ratio:.2f}" if ratio is not None else "N/A"
        ax.set_title(f"Unaffected Mask Overlay\nLateral Pillar Ratio: {ratio_txt}")
        ax.axis('off')

    def _plot_axes(self, ax):
        """Plot 3: Original radiograph with major/minor axes"""
        ax.imshow(self.data['original_image'], cmap='gray')

        # Overlay affected mask
        mask = self.data['affected_mask']
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 0, 1, 0.3]
        ax.imshow(overlay)

        # Plot major axis (red)
        (mx1, my1), (mx2, my2) = self.data['aff_major_axis']
        ax.plot([mx1, mx2], [my1, my2], 'r-', linewidth=2, label='Major Axis')

        # Plot minor axis (cyan)
        (mix1, miy1), (mix2, miy2) = self.data['aff_minor_axis']
        ax.plot([mix1, mix2], [miy1, miy2], 'c-', linewidth=2, label='Minor Axis')

        ax.set_title("Major & Minor Axes")
        ax.legend(loc='best')
        ax.axis('off')

    def _plot_deformity(self, ax):
        """Plot 4: Original + both masks with deformity index"""
        ax.imshow(self.data['original_image'], cmap='gray')

        # Overlay both masks
        aff_mask = self.data['affected_mask']
        unaff_mask = self.data['transformed_unaff_mask']

        # Create combined RGBA overlay
        overlay = np.zeros((*aff_mask.shape, 4))
        overlay[aff_mask] = [0, 0, 1, 0.3]  # Blue - affected
        overlay[unaff_mask] = [0, 1, 0, 0.3]  # Green - unaffected

        ax.imshow(overlay)

        # Add deformity index to title
        di = self.analyzer.di_measurements.get('deformity_index', None)
        di_txt = f"{di:.2f}" if di is not None else "N/A"
        ax.set_title(f"Deformity Visualization\nDeformity Index: {di_txt}")
        ax.axis('off')

    def visualize_quad(self):
        """Create a 1x4 subplot visualization"""
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Patient {self.data['patient_id']} - {self.data['timepoint']}", fontsize=16)

        self._plot_affected_only(axes[0])
        self._plot_unaffected_overlay(axes[1])
        self._plot_axes(axes[2])
        self._plot_deformity(axes[3])

        plt.tight_layout()
        quad_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_quad_vis.png"
        plt.savefig(quad_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return quad_path