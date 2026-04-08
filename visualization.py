
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from skimage import measure
from measurement import DIMeasurement

# ------------------------------
# Measurement-level visualizers
# ------------------------------
class MeasurementVisualizer:
    def __init__(self, analyzer, output_folder):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _apply_zoom(self, ax, mask, margin=60):
        """
        Zoom the axis to the bounding box of a boolean mask.
        Keeps aspect equal and uses image coordinates (origin='upper').
        """
        if mask is None:
            return
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(max_y + margin, min_y - margin)  # invert y for image coords
        ax.set_aspect('equal')

    def _draw_mask_outline(self, ax, mask, color):
        """Draw outline of a mask."""
        if mask is None:
            return
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    def _draw_pillar_divisions(self, ax, mask, laterality, color):
        """Draw vertical lines dividing pillars with labels."""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x

        # Pillar boundaries at 25% and 75% of width
        div1 = min_x + 0.25 * width
        div2 = min_x + 0.75 * width
        ax.axvline(x=div1, color=color, linestyle='--', alpha=0.7)
        ax.axvline(x=div2, color=color, linestyle='--', alpha=0.7)

        mid_y = (min_y + max_y) / 2
        if laterality == 'R':
            ax.text(div1 - 0.05 * width, mid_y, "Lateral", ha='right', va='center', color=color)
            ax.text((div1 + div2) / 2, mid_y, "Middle", ha='center', va='center', color=color)
            ax.text(div2 + 0.05 * width, mid_y, "Medial", ha='left', va='center', color=color)
        else:
            ax.text(div1 - 0.05 * width, mid_y, "Medial", ha='right', va='center', color=color)
            ax.text((div1 + div2) / 2, mid_y, "Middle", ha='center', va='center', color=color)
            ax.text(div2 + 0.05 * width, mid_y, "Lateral", ha='left', va='center', color=color)

    def _draw_bounding_box(self, ax, mask, color):
        """Draw bounding box around mask."""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
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
        """Draw width and height dimensions next to the bounding box."""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        # Width line above box
        mid_y = min_y - height * 0.05
        ax.plot([min_x, max_x], [mid_y, mid_y], color='gray', linewidth=1)
        ax.text((min_x + max_x) / 2, mid_y, f"W: {width:.1f}", ha='center', va='bottom', color='gray')

        # Height line to the right of box
        mid_x = max_x + width * 0.05
        ax.plot([mid_x, mid_x], [min_y, max_y], color='gray', linewidth=1)
        ax.text(mid_x, (min_y + max_y) / 2, f"H: {height:.1f}", ha='left', va='center', color='gray', rotation=90)

    def _draw_label(self, ax, text, color, x, y):
        """Draw a text label at normalized (axes) coordinates."""
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            color=color,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    def visualize_di(self):
        """Visualize DI measurement with landmark-aligned masks (zoomed to union)."""
        di_data = self.analyzer.di_measurements
        aff_mask = di_data.get('aff_padded', None)
        unaff_mask = di_data.get('unaff_padded', None)
        if aff_mask is None or unaff_mask is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        # Zoom to union of both masks
        union = np.logical_or(aff_mask, unaff_mask)
        self._apply_zoom(ax, union, margin=60)

        # Add labels
        self._draw_label(ax, "Affected", 'red', 0.05, 0.95)
        self._draw_label(ax, "Unaffected", 'lime', 0.05, 0.90)

        # Add landmark markers
        aff_landmark = DIMeasurement.find_landmark(aff_mask, self.data['affected_laterality'])
        unaff_landmark = DIMeasurement.find_landmark(unaff_mask, self.data['affected_laterality'])
        ax.scatter(aff_landmark[0], aff_landmark[1], s=80, c='red', marker='o', edgecolor='white')
        ax.scatter(unaff_landmark[0], unaff_landmark[1], s=80, c='lime', marker='o', edgecolor='white')

        # Optionally draw neck mask outlines when available
        aff_neck  = self.data.get('affected_neck_mask')
        unaff_neck = self.data.get('transformed_unaff_neck_mask')
        if aff_neck is not None and aff_neck.any():
            self._draw_mask_outline(ax, aff_neck, 'salmon')
        if unaff_neck is not None and unaff_neck.any():
            self._draw_mask_outline(ax, unaff_neck, 'lightgreen')

        # Title with measurements
        icp_err = di_data.get('icp_registration_error', float('nan'))
        icp_txt = f"{icp_err:.2f}px" if icp_err == icp_err else "fallback"  # nan check
        mode_txt = di_data.get('alignment_mode', 'head_only_icp')
        title = (
            f"Patient {self.data['patient_id']} - {self.data['timepoint']}\n"
            f"DI: {di_data.get('deformity_index', float('nan')):.2f}, "
            f"ΔH: {di_data.get('deltaH', float('nan')):.1f}, "
            f"ΔW: {di_data.get('deltaW', float('nan')):.1f}, "
            f"Unaff Diam: {di_data.get('unaff_diameter', float('nan')):.1f}, "
            f"ICP err: {icp_txt}, Mode: {mode_txt}"
        )
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Save and close
        di_folder = self.output_folder / "di_viz"
        di_folder.mkdir(exist_ok=True)
        save_path = di_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_di.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path

    def visualize_lateral_pillar(self):
        """Visualize lateral pillars with thirds division and COM markers (zoomed)."""
        aff_mask = self.analyzer.rotated_aff_mask
        unaff_mask = self.analyzer.rotated_unaff_mask
        pillar_data = self.analyzer.pillar_measurements

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        # Labels
        self._draw_label(ax, "Affected", 'red', 0.05, 0.95)
        self._draw_label(ax, "Unaffected", 'lime', 0.05, 0.90)

        # Pillar divisions
        self._draw_pillar_divisions(ax, aff_mask, self.data['affected_laterality'], 'red')
        self._draw_pillar_divisions(ax, unaff_mask, self.data['affected_laterality'], 'lime')

        # COM
        aff_com = self.analyzer.rotated_aff_com
        unaff_com = self.analyzer.rotated_unaff_com
        ax.scatter(aff_com[0], aff_com[1], s=100, c='red', marker='x', linewidth=2)
        ax.scatter(unaff_com[0], unaff_com[1], s=100, c='lime', marker='x', linewidth=2)
        ax.text(aff_com[0], aff_com[1] + 10, 'Aff COM', color='red', ha='center', va='bottom', fontsize=10)
        ax.text(unaff_com[0], unaff_com[1] + 10, 'Unaff COM', color='lime', ha='center', va='bottom', fontsize=10)

        # Title
        title = (
            f"Patient {self.data['patient_id']} - {self.data['timepoint']}\n"
            f"Lat Ratio: {pillar_data.get('ratio_lateral_avg', float('nan')):.2f}, "
            f"Med Ratio: {pillar_data.get('ratio_medial_avg', float('nan')):.2f}, "
            f"Herring: {pillar_data.get('herring_class', 'N/A')}"
        )
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Zoom to union for shared context
        union = np.logical_or(aff_mask, unaff_mask)
        self._apply_zoom(ax, union, margin=60)

        # Save and close
        lateral_folder = self.output_folder / "lateral_pillar_viz"
        lateral_folder.mkdir(exist_ok=True)
        save_path = lateral_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_pillar.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path

    def visualize_eq(self):
        """Visualize EQ with bounding boxes and dimensions (zoomed)."""
        aff_mask = self.analyzer.rotated_aff_mask
        unaff_mask = self.analyzer.rotated_unaff_mask
        eq_data = self.analyzer.eq_measurements

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # Draw masks
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        # Labels
        self._draw_label(ax, "Affected", 'red', 0.05, 0.95)
        self._draw_label(ax, "Unaffected", 'lime', 0.05, 0.90)

        # Bounding boxes + dimensions
        self._draw_bounding_box(ax, aff_mask, 'red')
        self._draw_bounding_box(ax, unaff_mask, 'lime')
        self._draw_dimensions(ax, aff_mask, 'red')
        self._draw_dimensions(ax, unaff_mask, 'lime')

        # Title
        title = (
            f"Patient {self.data['patient_id']} - {self.data['timepoint']}\n"
            f"EQ Ratio: {eq_data.get('eq_ratio', float('nan')):.2f}, "
            f"Aff EQ: {eq_data.get('aff_eq', float('nan')):.2f}, "
            f"Unaff EQ: {eq_data.get('unaff_eq', float('nan')):.2f}"
        )
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Zoom to union (so both boxes fully visible)
        union = np.logical_or(aff_mask, unaff_mask)
        self._apply_zoom(ax, union, margin=60)

        # Save and close
        eq_folder = self.output_folder / "eq_viz"
        eq_folder.mkdir(exist_ok=True)
        save_path = eq_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_eq.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path


# --------------------------------------------
# Final composite visualizer (with zooms)
# --------------------------------------------
class FinalMeasurementVisualizer:
    def __init__(self, analyzer, img_folder, output_folder, zoom_margin=20):
        self.analyzer = analyzer
        self.data = analyzer.data
        self.output_folder = Path(output_folder) / "final viz"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.img_folder = Path(img_folder)
        self.original_image = self._load_original_image()
        self.unaffected_image = self._load_unaffected_image()

        # Masks for zooming
        self.aff_mask = (
            self.data.get('affected_mask')
        )
        self.unaff_mask = (
            self.data.get('unaffected_mask')
        )
        self.zoom_margin = zoom_margin

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
        if mask is None:
            return
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

    @staticmethod
    def _union_bbox(masks):
        """Compute the bounding box of the union of one or more boolean masks."""
        xs, ys = [], []
        for m in masks:
            if m is None:
                continue
            coords = np.argwhere(m)
            if coords.size == 0:
                continue
            ys.extend(coords[:, 0])
            xs.extend(coords[:, 1])
        if not xs or not ys:
            return None
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, max_x, min_y, max_y)

    def _apply_zoom(self, ax, masks_or_mask, margin=100):
        """Set xlim/ylim to the bounding box of one mask or a list of masks."""
        margin = self.zoom_margin if margin is None else margin
        if isinstance(masks_or_mask, (list, tuple)):
            bbox = self._union_bbox(masks_or_mask)
        else:
            bbox = self._union_bbox([masks_or_mask])
        if bbox is None:
            return
        min_x, max_x, min_y, max_y = bbox
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(max_y + margin, min_y - margin)  # invert y for image coords
        ax.set_aspect('equal')

    # --------- small helpers for line lengths & labels ----------
    def _line_length(self, p1, p2):
        """Return Euclidean length between two (x,y) points."""
        if p1 is None or p2 is None:
            return 0.0
        return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    def _annotate_line_length(self, ax, p1, p2, text, color='white', offset=8):
        """
        Write a length label near the midpoint of a line segment.
        offset: pixel shift along the normal for readability.
        """
        if p1 is None or p2 is None:
            return
        mx = (p1[0] + p2[0]) / 2.0
        my = (p1[1] + p2[1]) / 2.0
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        L = np.hypot(dx, dy)
        if L < 1e-6:
            return
        # normal vector for label offset
        nx, ny = -dy / L, dx / L
        ax.text(mx + nx * offset, my + ny * offset, text,
                color=color, fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # ---------------- individual panels (with zoom) ----------------
    def _plot_affected_original(self, ax):
        ax.imshow(self.original_image, cmap='gray', interpolation='nearest')
        self._draw_mask_outline(ax, self.aff_mask, color='red')
        ax.set_title("Affected Femoral Head (Original)")
        ax.axis('off')

    def _plot_unaffected_original(self, ax):
        ax.imshow(self.unaffected_image, cmap='gray', interpolation='nearest')
        self._draw_mask_outline(ax, self.unaff_mask, color='lime')
        ax.set_title("Unaffected Femoral Head (Original)")
        ax.axis('off')

    def _plot_lateral_pillar(self, ax):
        ax.imshow(self.original_image, cmap='gray', interpolation='nearest')
        self._draw_mask_outline(ax, self.aff_mask, color='red')

        # Draw lines perpendicular to major axis at quarter points (optional visuals)
        aff_mask = self.aff_mask
        coords = np.argwhere(aff_mask) if aff_mask is not None else np.empty((0, 2))
        if coords.size > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            major_axis = self.data.get('aff_major_axis')
            if major_axis:
                (x1, y1), (x2, y2) = major_axis
                dx, dy = x2 - x1, y2 - y1
                length = np.hypot(dx, dy)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    vx, vy = -uy, ux
                    quarter1 = (x1 + 0.25 * dx, y1 + 0.25 * dy)
                    quarter3 = (x1 + 0.75 * dx, y1 + 0.75 * dy)
                    for (px, py) in [quarter1, quarter3]:
                        top_x = px - vx * abs(max_x - min_x) * 0.5
                        top_y = py - vy * abs(max_y - min_y) * 0.5
                        bot_x = px + vx * abs(max_x - min_x) * 0.5
                        bot_y = py + vy * abs(max_y - min_y) * 0.5
                        ax.plot([top_x, bot_x], [top_y, bot_y], color='blue', linewidth=2, linestyle='-')

        ratio = self.analyzer.pillar_measurements.get('ratio_lateral_avg', None)
        ratio_txt = f"{ratio:.2f}" if ratio is not None else "N/A"
        classification = self.analyzer.pillar_measurements.get('herring_class', None)
        classification_txt = classification if classification is not None else "N/A"
        ax.set_title(f"Lateral Pillar Comparison\nLateral Pillar Ratio: {ratio_txt} ({classification_txt})")
        self._apply_zoom(ax, self.aff_mask)
        ax.axis('off')

    def _plot_eq(self, ax):
        """Epiphyseal quotient panel with major axis (width) and greatest ⟂ height."""
        ax.imshow(self.original_image, cmap='gray', interpolation='nearest')
        self._draw_mask_outline(ax, self.aff_mask, color='red')

        # 1) Major axis: this is our 'width' line
        major = self.data.get('aff_major_axis')
        if major:
            (x1, y1), (x2, y2) = major
            ax.plot([x1, x2], [y1, y2], color='white', linewidth=2, alpha=0.9, label='Major Axis')
            width_len_px = self._line_length((x1, y1), (x2, y2))
            #self._annotate_line_length(ax, (x1, y1), (x2, y2), f"Width: {width_len_px:.1f}px", color='white')

        # 2) Greatest height perpendicular to major axis
        height_line = self.data.get('aff_max_height_line')  # (p_top, p_bottom)
        if height_line and height_line[0] is not None and height_line[1] is not None:
            (hx1, hy1), (hx2, hy2) = height_line
            ax.plot([hx1, hx2], [hy1, hy2], color='cyan', linewidth=2, alpha=0.9, label='Max ⟂ Height')
            height_len_px = self._line_length((hx1, hy1), (hx2, hy2))
            #self._annotate_line_length(ax, (hx1, hy1), (hx2, hy2), f"Height: {height_len_px:.1f}px", color='cyan')

        # Optional: keep minor axis (existing) for reference
        #if self.data.get('aff_minor_axis'):
        #    (mx1, my1), (mx2, my2) = self.data['aff_minor_axis']
        #    ax.plot([mx1, mx2], [my1, my2], color='yellow', linewidth=1.5, alpha=0.6, label='Minor Axis')

        # EQ numbers
        unaffeq = self.analyzer.eq_measurements.get('unaff_eq', None)
        affeq   = self.analyzer.eq_measurements.get('aff_eq', None)
        eq_ratio = self.analyzer.eq_measurements.get('eq_ratio', None)
        unaffeq_txt = f"{unaffeq:.2f}" if unaffeq is not None else "N/A"
        affeq_txt   = f"{affeq:.2f}" if affeq is not None else "N/A"
        eq_ratio_txt = f"{eq_ratio:.2f}" if eq_ratio is not None else "N/A"
        ax.set_title(f"Epiphyseal Quotient\nUnaffected EI: {unaffeq_txt}\nAffected EI: {affeq_txt}\nEQ: {eq_ratio_txt}")

        # Zoom and hide axes
        self._apply_zoom(ax, self.aff_mask)
        ax.axis('off')

    def _plot_deformity(self, ax):
        """Deformity visualization with both masks; zoom to union bbox."""
        di_data = self.analyzer.di_measurements
        aff_mask = di_data.get('aff_padded', self.aff_mask)
        unaff_mask = di_data.get('unaff_padded', self.unaff_mask)

        ax.imshow(self.original_image, cmap='gray', interpolation='nearest')
        self._draw_mask_outline(ax, aff_mask, 'red')
        self._draw_mask_outline(ax, unaff_mask, 'lime')

        di = self.analyzer.di_measurements.get('deformity_index', None)
        di_txt = f"{di:.2f}" if di is not None else "N/A"
        ax.set_title(f"Deformity Visualization\nDeformity Index: {di_txt}")

        # Zoom to union so both outlines are visible
        self._apply_zoom(ax, [aff_mask, unaff_mask])
        ax.axis('off')

    # ---------------- composite figure ----------------
    def visualize_quad(self):
        """
        Create 2x3 grid:
        - Top row: affected original (zoomed), unaffected original (zoomed), empty.
        - Bottom row: lateral pillar (zoomed), EQ (zoomed), deformity (zoomed to union).
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Patient {self.data['patient_id']} - {self.data['timepoint']}", fontsize=16)

        # Top row
        self._plot_affected_original(axes[0, 0])
        self._plot_unaffected_original(axes[0, 1])
        axes[0, 2].axis('off')

        # Bottom row
        self._plot_lateral_pillar(axes[1, 0])
        self._plot_eq(axes[1, 1])
        self._plot_deformity(axes[1, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # space for suptitle
        quad_path = self.output_folder / f"{self.data['patient_id']}_{self.data['timepoint']}_quad_vis.png"
        plt.savefig(quad_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
