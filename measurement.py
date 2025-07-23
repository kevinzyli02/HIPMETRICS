import numpy as np
import cv2
from abc import ABC, abstractmethod


class BaseMeasurement(ABC):
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.data = analyzer.data

    @abstractmethod
    def calculate(self):
        pass

class PillarMeasurement(BaseMeasurement):
    def _calculate_pillar_heights(self, mask, laterality):
        """
        Calculate maximum height, average height, and width for the three
        anatomical pillars (lateral, middle, medial) according to Herring classification.

        Args:
            mask (ndarray): 2D binary mask of the femoral head
            laterality (str): Laterality of the hip, either 'R' (right) or 'L' (left)

        Returns:
            list[float]: List of 9 values ordered as:
                [lateral_max, lateral_avg, lateral_width,
                 middle_max, middle_avg, middle_width,
                 medial_max, medial_avg, medial_width]
        """
        try:
            # Get bounding box of non-zero region
            coords = np.argwhere(mask)
            if coords.size == 0:
                return [0.0] * 9  # Return 9 zeros for empty mask

            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            width = max_x - min_x

            # Define pillar boundaries (25% medial, 50% middle, 25% lateral)
            left_bound = min_x
            middle_start = min_x + 0.25 * width
            middle_end = min_x + 0.75 * width
            right_bound = max_x

            # Create pillar segments
            pillars = {
                'medial': (left_bound, middle_start),
                'middle': (middle_start, middle_end),
                'lateral': (middle_end, right_bound)
            }

            # Determine pillar order based on laterality
            if laterality == 'R':
                pillar_order = ['lateral', 'middle', 'medial']
            else:  # 'L'
                pillar_order = ['medial', 'middle', 'lateral']

            # Measure heights and width for each pillar
            results = []
            for name in pillar_order:
                start, end = pillars[name]
                start_col = int(np.floor(start))
                end_col = int(np.ceil(end))

                # Ensure valid column indices
                start_col = max(start_col, 0)
                end_col = min(end_col, mask.shape[1] - 1)

                # Skip if invalid segment
                if start_col >= mask.shape[1] or end_col < 0 or start_col > end_col:
                    results.extend([0.0, 0.0, 0.0])
                    continue

                pillar_mask = mask[:, start_col:end_col + 1]
                pillar_width = end_col - start_col + 1

                heights = []
                for col in range(pillar_mask.shape[1]):
                    col_mask = pillar_mask[:, col]
                    if np.any(col_mask):
                        y_indices = np.where(col_mask)[0]
                        height = y_indices.max() - y_indices.min() + 1
                        heights.append(float(height))

                max_height = np.max(heights) if heights else 0.0
                avg_height = np.mean(heights) if heights else 0.0
                results.extend([max_height, avg_height, float(pillar_width)])

            return results

        except Exception as e:
            print(f"Error in _calculate_pillar_heights: {str(e)}")
            import traceback
            traceback.print_exc()
            return [0.0] * 9  # Always return 9 values

    def _determine_herring_class(self, aff_heights, unaff_heights):
        """
        Determine Herring classification (A, B, C) based on lateral pillar height ratio

        Classification criteria:
        - A: Lateral pillar height >= 95% of original height
        - B: Lateral pillar height between 50-95% of original height
        - C: Lateral pillar height < 50% of original height
        - B/C border: Special case (not auto-classified)

        Args:
            aff_heights (list): 9 measurements for affected hip
            unaff_heights (list): 9 measurements for unaffected hip

        Returns:
            str: Herring classification (A, B, C, or B/C border)
        """
        # Extract lateral pillar max heights (index 0 in both lists)
        aff_lat_max = aff_heights[0]
        unaff_lat_max = unaff_heights[0]

        # Handle edge cases
        if unaff_lat_max == 0:
            return "Undefined (reference height=0)"

        # Calculate height ratio
        height_ratio = aff_lat_max / unaff_lat_max

        # Determine classification
        if height_ratio >= 0.95:
            return "A"
        elif height_ratio >= 0.5:
            return "B"
        else:
            return "C"

    def calculate(self):
        """Calculate pillar measurements and determine Herring classification"""
        try:
            # Measure for affected and unaffected masks
            aff_heights = self._calculate_pillar_heights(
                self.analyzer.rotated_aff_mask, self.data['affected_laterality']
            )
            unaff_heights = self._calculate_pillar_heights(
                self.analyzer.rotated_unaff_mask, self.data['unaffected_laterality']
            )

            # Print debug information
            #print(f"Affected heights: {aff_heights}")
            #print(f"Unaffected heights: {unaff_heights}")

            # Calculate ratios (avoid division by zero)
            ratios = []
            for aff_val, unaff_val in zip(aff_heights, unaff_heights):
                ratio = aff_val / unaff_val if unaff_val != 0 else float('nan')
                ratios.append(ratio)

            # Store measurements
            pillar_types = [
                'lateral_max', 'lateral_avg', 'lateral_width',
                'middle_max', 'middle_avg', 'middle_width',
                'medial_max', 'medial_avg', 'medial_width'
            ]

            self.analyzer.pillar_measurements = {
                'aff_' + k: v for k, v in zip(pillar_types, aff_heights)
            }
            self.analyzer.pillar_measurements.update({
                'unaff_' + k: v for k, v in zip(pillar_types, unaff_heights)
            })
            self.analyzer.pillar_measurements.update({
                'ratio_' + k: v for k, v in zip(pillar_types, ratios)
            })

            # Add Herring classification
            herring_class = self._determine_herring_class(aff_heights, unaff_heights)
            self.analyzer.herring_classification = herring_class
            self.analyzer.pillar_measurements['herring_class'] = herring_class

        except Exception as e:
            print(f"Error in PillarMeasurement.calculate: {str(e)}")
            import traceback
            traceback.print_exc()

class EQMeasurement(BaseMeasurement):
    def _calculate_epiphyseal_quotient(self, mask):
        """Calculate height/width ratio for femoral head mask"""
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

    def calculate(self):
        """Calculate EQ (height/width) for both femoral heads"""
        # Calculate EQ for affected head
        aff_eq, aff_width, aff_height = self._calculate_epiphyseal_quotient(
            self.analyzer.rotated_aff_mask
        )
        unaff_eq, unaff_width, unaff_height = self._calculate_epiphyseal_quotient(
            self.analyzer.rotated_unaff_mask
        )

        # Calculate ratio
        eq_ratio = aff_eq / unaff_eq if unaff_eq != 0 else float('nan')

        # Store measurements
        self.analyzer.eq_measurements = {
            'aff_eq': aff_eq,
            'aff_width': aff_width,
            'aff_height': aff_height,
            'unaff_eq': unaff_eq,
            'unaff_width': unaff_width,
            'unaff_height': unaff_height,
            'eq_ratio': eq_ratio
        }


class DIMeasurement(BaseMeasurement):
    @staticmethod
    def find_landmark(mask, laterality):
        """Finds the landmark point (lowest and most lateral) in a mask"""
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
        """Computes top/bottom profiles (per column) and left/right profiles (per row)"""
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

    def calculate(self):
        # Get masks and laterality
        affected_mask = self.data['affected_mask']
        transformed_unaff_mask = self.data['transformed_unaff_mask']
        affected_laterality = self.data['affected_laterality']
        unaff_width = self.analyzer.eq_measurements['unaff_width']

        # Find landmarks
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

        # Calculate max width difference
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
        self.analyzer.di_measurements = {
            'deformity_index': deformity_index,
            'deltaH': deltaH,
            'deltaW': deltaW,
            'unaff_diameter': unaff_width,
            'aff_padded': aff_padded,
            'unaff_padded': unaff_padded,
        }