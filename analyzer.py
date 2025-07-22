import numpy as np
from skimage.transform import rotate
from pathlib import Path
from measurement import PillarMeasurement, EQMeasurement, DIMeasurement
from visualization import FemoralHeadVisualizer


class FemoralHeadAnalyzer:
    def __init__(self, result_dict, images_folder, output_folder):
        self.data = result_dict
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.img_folder = Path(images_folder)

        # Initialize measurement modules
        self.pillar_measurement = PillarMeasurement(self)
        self.eq_measurement = EQMeasurement(self)
        self.di_measurement = DIMeasurement(self)
        self.visualizer = FemoralHeadVisualizer(self, output_folder)

        # Initialize results storage
        self.pillar_measurements = {}
        self.eq_measurements = {}
        self.di_measurements = {}
        self.rotated_aff_mask = None
        self.rotated_unaff_mask = None
        self.unaff_flipped = False

    # Rotation helper methods
    def _compute_rotation_angle(self, axis_endpoints):
        """Calculate rotation angle to make major axis horizontal"""
        (x1, y1), (x2, y2) = axis_endpoints
        dx, dy = x2 - x1, y2 - y1
        return np.degrees(np.arctan2(dy, dx))

    def _rotate_mask(self, mask, angle, center=None):
        """
        Rotate mask around a specified center
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
        """Rotate a point around a center by given angle in degrees"""
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
        """Rotate masks around center of major axis to make it horizontal"""
        # Calculate rotation angle
        angle = self._compute_rotation_angle(self.data['aff_major_axis'])

        # Get rotation centers
        aff_center = self._get_major_axis_center(self.data['aff_major_axis'])
        unaff_center = self._get_major_axis_center(self.data['trans_unaff_major_axis'])

        # Rotate both masks
        self.rotated_aff_mask = self._rotate_mask(
            self.data['affected_mask'], angle, center=aff_center
        )
        self.rotated_unaff_mask = self._rotate_mask(
            self.data['transformed_unaff_mask'], angle, center=unaff_center
        )

        # 180° rotation check for unaffected mask
        com_orig = self.data['com_unaff_trans']  # (x, y)
        com_orig_rot = self._rotate_point(com_orig, unaff_center, angle)
        com_rotated_180 = self._rotate_point(com_orig_rot, unaff_center, 180)

        # Calculate relative COM positions
        aff_com_vector = np.array(self.data['com_aff']) - np.array(aff_center)
        unaff_com_vector = np.array(com_orig_rot) - np.array(unaff_center)

        # Get vertical directions (y-components)
        aff_vertical_sign = np.sign(aff_com_vector[1])
        unaff_vertical_sign0 = np.sign(unaff_com_vector[1])

        # Flip if vertical orientations don't match
        if aff_vertical_sign != unaff_vertical_sign0:
            self.rotated_unaff_mask = self._rotate_mask(
                self.rotated_unaff_mask, 180, center=unaff_center
            )
            self.rotated_aff_mask = self._rotate_mask(
                self.rotated_aff_mask, 180, center=aff_center
            )
            self.unaff_flipped = True
        else:
            self.unaff_flipped = False

    def get_results(self):
        """Return all measurements as a dictionary"""
        result = {
            'patient_id': self.data['patient_id'],
            'timepoint': self.data['timepoint'],
            'affected_laterality': self.data['affected_laterality'],
            'unaffected_laterality': self.data['unaffected_laterality'],
            'lateral_ratio': self.pillar_measurements.get('ratio_lateral_max', None),
            'middle_ratio': self.pillar_measurements.get('ratio_middle_max', None),
            'medial_ratio': self.pillar_measurements.get('ratio_medial_max', None),
            'aff_lateral_max': self.pillar_measurements.get('aff_lateral_max', None),
            'aff_lateral_avg': self.pillar_measurements.get('aff_lateral_avg', None),
            'aff_middle_max': self.pillar_measurements.get('aff_middle_max', None),
            'aff_middle_avg': self.pillar_measurements.get('aff_middle_avg', None),
            'aff_medial_max': self.pillar_measurements.get('aff_medial_max', None),
            'aff_medial_avg': self.pillar_measurements.get('aff_medial_avg', None),
            'unaff_lateral_max': self.pillar_measurements.get('unaff_lateral_max', None),
            'unaff_lateral_avg': self.pillar_measurements.get('unaff_lateral_avg', None),
            'unaff_middle_max': self.pillar_measurements.get('unaff_middle_max', None),
            'unaff_middle_avg': self.pillar_measurements.get('unaff_middle_avg', None),
            'unaff_medial_max': self.pillar_measurements.get('unaff_medial_max', None),
            'unaff_medial_avg': self.pillar_measurements.get('unaff_medial_avg', None),
            'eq_ratio': self.eq_measurements.get('eq_ratio', None),
            'aff_eq': self.eq_measurements.get('aff_eq', None),
            'aff_width': self.eq_measurements.get('aff_width', None),
            'aff_height': self.eq_measurements.get('aff_height', None),
            'unaff_eq': self.eq_measurements.get('unaff_eq', None),
            'unaff_width': self.eq_measurements.get('unaff_width', None),
            'unaff_height': self.eq_measurements.get('unaff_height', None),
            'deformity_index': self.di_measurements.get('deformity_index', None),
            'deltaH': self.di_measurements.get('deltaH', None),
            'deltaW': self.di_measurements.get('deltaW', None),
            'unaff_diameter': self.di_measurements.get('unaff_diameter', None),
        }
        return result

    def process(self):
        """Complete processing pipeline for a single patient/timepoint"""
        # Perform alignment
        self.align_major_axis()

        # Run measurements
        self.pillar_measurement.calculate()
        self.eq_measurement.calculate()
        self.di_measurement.calculate()

        # Generate visualization
        vis_path = self.visualizer.visualize_quad()

        # Return visualization path and results
        return vis_path, self.get_results()