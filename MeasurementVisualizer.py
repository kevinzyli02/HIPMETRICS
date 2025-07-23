import cv2
import numpy as np


class MeasurementVisualizer:
    def __init__(self, visualize=False):
        self.visualize = visualize
        self.visualizations = {}

    def visualize_pillars(self, mask, laterality, boundaries, pillar_order):
        """Visualize Lateral Pillar measurement"""
        if not self.visualize:
            return None

        # Create RGB image
        vis_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis_img[mask > 0] = [200, 200, 200]  # Gray mask

        # Define colors for each pillar
        colors = {
            'lateral': (0, 0, 255),  # Red
            'middle': (0, 255, 0),  # Green
            'medial': (255, 0, 0)  # Blue
        }

        # Draw boundaries
        for x in [boundaries['medial'][1], boundaries['middle'][1]]:
            x = int(x)
            if 0 <= x < vis_img.shape[1]:
                cv2.line(vis_img, (x, 0), (x, vis_img.shape[0]), (0, 255, 255), 2)  # Yellow lines

        # Draw pillar regions
        for name in pillar_order:
            start, end = boundaries[name]
            start_col = int(start)
            end_col = int(end)
            color = colors[name]

            # Draw pillar region
            if start_col < end_col:
                region = vis_img[:, start_col:end_col].copy()
                region[:, :, :] = color
                overlay = vis_img.copy()
                overlay[:, start_col:end_col] = region
                cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

        # Add legend and laterality
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, f"Lateral Pillar Classification ({laterality})", (10, 30), font, 0.7, (255, 255, 0), 2)

        y_pos = 70
        for name in pillar_order:
            color = colors[name]
            cv2.putText(vis_img, f"{name.capitalize()} Pillar", (10, y_pos), font, 0.6, color, 2)
            y_pos += 30

        # Add boundary percentages
        cv2.putText(vis_img, "25%", (int(boundaries['medial'][1]) - 20, 30), font, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_img, "75%", (int(boundaries['middle'][1]) - 20, 30), font, 0.5, (0, 255, 255), 1)

        return vis_img

    def visualize_eq(self, mask, bounding_box):
        """Visualize Epiphyseal Quotient measurement"""
        if not self.visualize:
            return None

        # Create RGB image
        vis_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis_img[mask > 0] = [200, 200, 200]  # Gray mask

        # Draw bounding box
        x, y, w, h = bounding_box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box

        # Add measurement annotation
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, "Epiphyseal Quotient Measurement", (10, 30), font, 0.7, (255, 255, 0), 2)
        cv2.putText(vis_img, f"Width: {w}, Height: {h}", (10, 70), font, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_img, f"EQ: {h / w:.2f}" if w > 0 else "EQ: N/A", (10, 100), font, 0.5, (0, 255, 0), 1)

        return vis_img

    def visualize_di(self, aff_padded, unaff_padded, aff_landmark, unaff_landmark, deltaH, deltaW, di):
        """Visualize Deformity Index measurement"""
        if not self.visualize:
            return None

        # Create RGB image with both masks
        vis_img = np.zeros((aff_padded.shape[0], aff_padded.shape[1], 3), dtype=np.uint8)
        vis_img[aff_padded > 0] = [255, 0, 0]  # Affected in red
        vis_img[unaff_padded > 0] = [0, 255, 0]  # Unaffected in green

        # Draw landmarks
        cv2.circle(vis_img, aff_landmark, 8, (0, 0, 255), -1)  # Affected landmark: blue
        cv2.circle(vis_img, unaff_landmark, 8, (255, 255, 0), -1)  # Unaffected landmark: cyan

        # Add measurement annotation
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, "Deformity Index Measurement", (10, 30), font, 0.7, (255, 255, 0), 2)
        cv2.putText(vis_img, f"ΔH: {deltaH}, ΔW: {deltaW}", (10, 70), font, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_img, f"DI: {di:.2f}", (10, 100), font, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_img, "Red: Affected, Green: Unaffected", (10, 130), font, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, "Blue: Aff Landmark, Cyan: Unaff Landmark", (10, 160), font, 0.5, (255, 255, 255), 1)

        return vis_img