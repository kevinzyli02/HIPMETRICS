import numpy as np
from skimage.transform import rotate
from pathlib import Path
from measurement import PillarMeasurement, EQMeasurement, DIMeasurement
from visualization import FinalMeasurementVisualizer, MeasurementVisualizer


class FemoralHeadAnalyzer:
    def __init__(self, result_dict, images_folder, output_folder):
        self.rotated_aff_com = None
        self.rotated_unaff_com = None
        self.data = result_dict
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.img_folder = Path(images_folder)

        # Initialize measurement modules
        self.pillar_measurement = PillarMeasurement(self)
        self.eq_measurement = EQMeasurement(self)
        self.di_measurement = DIMeasurement(self)
        self.visualizer = FinalMeasurementVisualizer(self, images_folder, output_folder)

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
    def _compute_com(self, mask):
        """Compute center of mass for a mask"""
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return (0, 0)
        return (np.mean(x_indices), np.mean(y_indices))

    def align_major_axis(self):
        """Rotate masks around center of major axis to make it horizontal and ensure COM is above the axis"""
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
        com_rotated_unaff = self._compute_com(self.rotated_unaff_mask)
        com_rotated_aff = self._compute_com(self.rotated_aff_mask)

        # Calculate 180 up and down COM positions
        com_rotated_unaff_180 = self._rotate_point(com_rotated_unaff, unaff_center, 180)
        com_rotated_aff_180 = self._rotate_point(com_rotated_aff, aff_center, 180)

        # Compare y-values and rotate 180° if needed
        if com_rotated_unaff_180[1] > com_rotated_unaff[1]:
            self.rotated_unaff_mask = self._rotate_mask(
                self.rotated_unaff_mask, 180, center=unaff_center
            )
            self.rotated_aff_mask = self._rotate_mask(
                self.rotated_aff_mask, 180, center=aff_center
            )
            self.rotated_unaff_com = com_rotated_unaff_180
            self.rotated_aff_com = com_rotated_aff_180
            self.unaff_flipped = True


        else:
            self.rotated_unaff_com = com_rotated_unaff
            self.rotated_aff_com = com_rotated_aff
            self.unaff_flipped = False

        # Troubleshooting info
        # print('unaffected side side')
        # print(f"com before and after rotation: {com_rotated_unaff[1]} {com_rotated_unaff_180[1]}")
        # print(f"did we flip: {self.unaff_flipped}")


    def get_results(self):
        """Return all measurements as a dictionary"""
        result = {
            'patient_id': self.data['patient_id'],
            'timepoint': self.data['timepoint'],
            'affected_laterality': self.data['affected_laterality'],
            'unaffected_laterality': self.data['unaffected_laterality'],
            'herring_class': self.pillar_measurements['herring_class'],
            'lateral_ratio': self.pillar_measurements.get('ratio_lateral_avg', None),
            'middle_ratio': self.pillar_measurements.get('ratio_middle_avg', None),
            'medial_ratio': self.pillar_measurements.get('ratio_medial_avg', None),
            'lateral_width_ratio': self.pillar_measurements.get('ratio_lateral_width_max', None),
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
           # 'com_affected':
           # 'major_axis_affected':self.
            #'minor_axis_affected':
            #'com_unaffected':
            #'major_axis_unaffected':
            #'minor_axis_unaffected':

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

        # Generate measurement visualizations
        meas_visualizer = MeasurementVisualizer(self, self.output_folder)
        meas_visualizer.visualize_di()
        meas_visualizer.visualize_lateral_pillar()
        meas_visualizer.visualize_eq()

        # Generate original visualization
        vis_path = self.visualizer.visualize_quad()

        return vis_path, self.get_results()