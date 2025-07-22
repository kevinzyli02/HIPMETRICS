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
        # Get bounding box of non-zero region
        coords = np.argwhere(mask)
        if coords.size == 0:
            return [0, 0, 0, 0, 0, 0]

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

    def calculate(self):
        """Calculate pillar heights for both masks and ratios"""
        # Measure heights for affected and unaffected masks
        aff_heights = self._calculate_pillar_heights(
            self.analyzer.rotated_aff_mask, self.data['affected_laterality']
        )
        unaff_heights = self._calculate_pillar_heights(
            self.analyzer.rotated_unaff_mask, self.data['unaffected_laterality']
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

        self.analyzer.pillar_measurements = {
            'aff_' + k: v for k, v in zip(pillar_types, aff_heights)
        }
        self.analyzer.pillar_measurements.update({
            'unaff_' + k: v for k, v in zip(pillar_types, unaff_heights)
        })
        self.analyzer.pillar_measurements.update({
            'ratio_' + k: v for k, v in zip(pillar_types, ratios)
        })


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
        self.analyzer.di_measurements = {
            'deformity_index': deformity_index,
            'deltaH': deltaH,
            'deltaW': deltaW,
            'unaff_diameter': unaff_width,
        }

class PillarMeasurement(BaseMeasurement):
    def _calculate_pillar_heights(self, mask, laterality):
        # Get bounding box of non-zero region
        coords = np.argwhere(mask)
        if coords.size == 0:
            return [0, 0, 0, 0, 0, 0]

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

    def calculate(self):
        """Calculate pillar heights for both masks and ratios"""
        # Measure heights for affected and unaffected masks
        aff_heights = self._calculate_pillar_heights(
            self.analyzer.rotated_aff_mask, self.data['affected_laterality']
        )
        unaff_heights = self._calculate_pillar_heights(
            self.analyzer.rotated_unaff_mask, self.data['unaffected_laterality']
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

        self.analyzer.pillar_measurements = {
            'aff_' + k: v for k, v in zip(pillar_types, aff_heights)
        }
        self.analyzer.pillar_measurements.update({
            'unaff_' + k: v for k, v in zip(pillar_types, unaff_heights)
        })
        self.analyzer.pillar_measurements.update({
            'ratio_' + k: v for k, v in zip(pillar_types, ratios)
        })


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
        self.analyzer.di_measurements = {
            'deformity_index': deformity_index,
            'deltaH': deltaH,
            'deltaW': deltaW,
            'unaff_diameter': unaff_width,
        }