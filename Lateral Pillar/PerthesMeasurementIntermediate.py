import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse
import matplotlib.colors as mcolors
import os


class PerthesMeasurements:
    def __init__(self, aligned_results):
        """
        Initialize with aligned femoral head results.

        Args:
            aligned_results (list): List of dictionaries from femoral head alignment
        """
        self.aligned_results = aligned_results
        self.measurements = []

    def calculate_lateral_pillar(self, mask, major_axis_endpoints):
        """
        Calculate lateral pillar height from femoral head mask and major axis.

        Args:
            mask (ndarray): Binary mask of femoral head
            major_axis_endpoints (tuple): ((x1, y1), (x2, y2)) of major axis endpoints

        Returns:
            tuple: (height, lateral_points, lateral_region)
        """
        # Extract axis points
        p1, p2 = major_axis_endpoints
        axis_vector = np.array(p2) - np.array(p1)
        axis_length = np.linalg.norm(axis_vector)
        u_axis = axis_vector / axis_length

        # Find convex hull points
        points = np.argwhere(mask)[:, [1, 0]]  # Convert to (x, y)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Project points onto major axis
        proj = np.dot(hull_points - p1, u_axis)

        # Identify lateral third (most lateral points)
        lateral_threshold = np.max(proj) - (axis_length / 3)
        lateral_mask = proj >= lateral_threshold
        lateral_points = hull_points[lateral_mask]

        if len(lateral_points) < 2:
            return 0.0, lateral_points, []

        # Create perpendicular vector
        u_perp = np.array([-u_axis[1], u_axis[0]])

        # Project lateral points onto perpendicular axis
        perp_proj = np.dot(lateral_points - lateral_points.mean(axis=0), u_perp)
        height = np.max(perp_proj) - np.min(perp_proj)

        # Find lateral region polygon
        lateral_region = []
        if len(lateral_points) > 2:
            # Find convex hull of lateral points
            try:
                lateral_hull = ConvexHull(lateral_points)
                lateral_region = lateral_points[lateral_hull.vertices]
            except:
                lateral_region = lateral_points

        return height, lateral_points, lateral_region

    def calculate_epiphyseal_quotient(self, mask):
        """
        Calculate Epiphyseal Quotient (EQ) for a femoral head mask.

        EQ = Height at pseudo minor axis / Width at pseudo major axis

        Args:
            mask (ndarray): Binary mask of femoral head

        Returns:
            tuple: (EQ, height, width, bbox_coords)
        """
        # Find all points in the mask
        points = np.argwhere(mask)
        if len(points) < 3:
            return 0.0, 0.0, 0.0, []

        # Convert to (x, y) format
        points_xy = points[:, [1, 0]]

        # Calculate bounding box dimensions
        min_x, min_y = np.min(points_xy, axis=0)
        max_x, max_y = np.max(points_xy, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        bbox_coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ]

        # Simple EQ calculation: height/width ratio
        if width > 0:
            eq = height / width
        else:
            eq = 0.0

        return eq, height, width, bbox_coords

    def calculate_all_measurements(self):
        """
        Calculate all Perthes measurements for all aligned results.

        Populates:
            self.measurements - List of measurement dictionaries
        """
        self.measurements = []

        for result in self.aligned_results:
            # Lateral Pillar measurements
            A_lateral, A_lat_points, A_lat_region = self.calculate_lateral_pillar(
                result['affected_mask'],
                result['aff_major_axis']
            )
            B_lateral, B_lat_points, B_lat_region = self.calculate_lateral_pillar(
                result['transformed_unaff_mask'],
                result['trans_unaff_major_axis']
            )
            LP_ratio = A_lateral / B_lateral if B_lateral > 0 else float('nan')

            # Epiphyseal Quotient measurements
            A_eq, A_height, A_width, A_bbox = self.calculate_epiphyseal_quotient(
                result['affected_mask']
            )
            B_eq, B_height, B_width, B_bbox = self.calculate_epiphyseal_quotient(
                result['unaffected_original_mask']  # Use original unaffected mask
            )

            # Deformity Index (1 - (A/B))
            DI = 1 - (A_eq / B_eq) if B_eq > 0 else float('nan')

            # Create measurement record
            measurement = {
                'patient_id': result['patient_id'],
                'timepoint': result['timepoint'],
                'affected_lateral': A_lateral,
                'unaffected_lateral': B_lateral,
                'LP_ratio': LP_ratio,
                'affected_EQ': A_eq,
                'unaffected_EQ': B_eq,
                'affected_height': A_height,
                'affected_width': A_width,
                'unaffected_height': B_height,
                'unaffected_width': B_width,
                'deformity_index': DI,
                'com_distance': result['dist'],
                # Visualization data
                'aff_lat_points': A_lat_points,
                'aff_lat_region': A_lat_region,
                'unaff_lat_points': B_lat_points,
                'unaff_lat_region': B_lat_region,
                'aff_bbox': A_bbox,
                'unaff_bbox': B_bbox,
                'result_data': result  # Reference to original result
            }
            self.measurements.append(measurement)

    def to_dataframe(self):
        """Convert measurements to pandas DataFrame"""
        return pd.DataFrame(self.measurements)

    def to_csv(self, file_path):
        """Save measurements to CSV file"""
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)

    def visualize_lateral_pillar(self, measurement, head_type='affected', ax=None, save_path=None):
        """
        Visualize lateral pillar measurement on femoral head.

        Args:
            measurement (dict): Single measurement dictionary
            head_type (str): 'affected' or 'unaffected'
            ax (matplotlib.axes.Axes): Existing axis to plot on
            save_path (str): Path to save visualization
        """
        # Get data based on head type
        if head_type == 'affected':
            mask_key = 'affected_mask'
            major_axis_key = 'aff_major_axis'
            lat_points = measurement['aff_lat_points']
            lat_region = measurement['aff_lat_region']
            height = measurement['affected_lateral']
        else:
            mask_key = 'transformed_unaff_mask'
            major_axis_key = 'trans_unaff_major_axis'
            lat_points = measurement['unaff_lat_points']
            lat_region = measurement['unaff_lat_region']
            height = measurement['unaffected_lateral']

        mask = measurement['result_data'][mask_key]
        major_axis = measurement['result_data'][major_axis_key]

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()

        # Create RGB representation of mask
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb_mask[mask] = [70, 130, 180]  # Steel blue

        # Plot mask
        ax.imshow(rgb_mask)

        # Plot convex hull points
        if len(lat_points) > 0:
            ax.scatter(lat_points[:, 0], lat_points[:, 1],
                       s=20, c='yellow', alpha=0.7, label='Lateral Points')

        # Plot lateral region
        if len(lat_region) > 0:
            poly = Polygon(lat_region, closed=True,
                           fill=True, alpha=0.3, color='orange')
            ax.add_patch(poly)

        # Plot major axis
        p1, p2 = major_axis
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                'r-', linewidth=2, label='Major Axis')

        # Calculate and plot lateral pillar height
        if height > 0 and len(lat_points) > 1:
            # Find min and max points in perpendicular direction
            mid_point = np.mean(lat_points, axis=0)
            u_perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
            u_perp /= np.linalg.norm(u_perp)

            perp_proj = np.dot(lat_points - mid_point, u_perp)
            min_idx = np.argmin(perp_proj)
            max_idx = np.argmax(perp_proj)

            # Plot height line
            ax.plot([lat_points[min_idx, 0], lat_points[max_idx, 0]],
                    [lat_points[min_idx, 1], lat_points[max_idx, 1]],
                    'g-', linewidth=3, label=f'Lateral Height: {height:.1f}px')

        # Configure plot
        ax.set_title(
            f"{head_type.capitalize()} Head Lateral Pillar\nPatient {measurement['patient_id']} - {measurement['timepoint']}")
        ax.legend()
        ax.axis('off')

        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return None
        else:
            return fig

    def visualize_epiphyseal_quotient(self, measurement, head_type='affected', ax=None, save_path=None):
        """
        Visualize epiphyseal quotient measurement on femoral head.

        Args:
            measurement (dict): Single measurement dictionary
            head_type (str): 'affected' or 'unaffected'
            ax (matplotlib.axes.Axes): Existing axis to plot on
            save_path (str): Path to save visualization
        """
        # Get data based on head type
        if head_type == 'affected':
            mask_key = 'affected_mask'
            bbox = measurement['aff_bbox']
            eq = measurement['affected_EQ']
            height = measurement['affected_height']
            width = measurement['affected_width']
        else:
            mask_key = 'unaffected_original_mask'
            bbox = measurement['unaff_bbox']
            eq = measurement['unaffected_EQ']
            height = measurement['unaffected_height']
            width = measurement['unaffected_width']

        mask = measurement['result_data'][mask_key]

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()

        # Create RGB representation of mask
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb_mask[mask] = [70, 130, 180]  # Steel blue

        # Plot mask
        ax.imshow(rgb_mask)

        # Plot bounding box
        if len(bbox) > 0:
            rect = Polygon(bbox, closed=True,
                           fill=False, linewidth=3,
                           edgecolor='red', linestyle='--')
            ax.add_patch(rect)

            # Add dimension labels
            min_x, min_y = bbox[0]
            max_x, max_y = bbox[2]
            ax.text(min_x + width / 2, min_y - 5, f"Width: {width:.1f}px",
                    color='red', fontsize=12, ha='center')
            ax.text(min_x - 10, min_y + height / 2, f"Height: {height:.1f}px",
                    color='red', fontsize=12, va='center', rotation=90)

        # Add EQ text
        ax.text(0.5, 0.95, f"EQ = {eq:.2f}",
                transform=ax.transAxes, color='white', fontsize=14,
                ha='center', bbox=dict(facecolor='black', alpha=0.7))

        # Configure plot
        ax.set_title(
            f"{head_type.capitalize()} Head Epiphyseal Quotient\nPatient {measurement['patient_id']} - {measurement['timepoint']}")
        ax.axis('off')

        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return None
        else:
            return fig

    def visualize_measurements(self, measurement, save_path=None):
        """
        Create comprehensive visualization of all measurements for one case.

        Args:
            measurement (dict): Single measurement dictionary
            save_path (str): Path to save visualization
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Perthes Measurements - Patient {measurement['patient_id']} - {measurement['timepoint']}",
                     fontsize=20)

        # Panel 1: Affected Lateral Pillar
        self.visualize_lateral_pillar(measurement, 'affected', ax=axs[0, 0])

        # Panel 2: Unaffected Lateral Pillar
        self.visualize_lateral_pillar(measurement, 'unaffected', ax=axs[0, 1])

        # Panel 3: Affected Epiphyseal Quotient
        self.visualize_epiphyseal_quotient(measurement, 'affected', ax=axs[1, 0])

        # Panel 4: Unaffected Epiphyseal Quotient
        self.visualize_epiphyseal_quotient(measurement, 'unaffected', ax=axs[1, 1])

        # Add summary text
        summary_text = (
            f"Lateral Pillar Ratio: {measurement['LP_ratio']:.2f}\n"
            f"Affected EQ: {measurement['affected_EQ']:.2f}\n"
            f"Unaffected EQ: {measurement['unaffected_EQ']:.2f}\n"
            f"Deformity Index: {measurement['deformity_index']:.2f}\n"
            f"COM Distance: {measurement['com_distance']:.1f}px"
        )
        fig.text(0.5, 0.05, summary_text, ha='center', fontsize=16,
                 bbox=dict(facecolor='lightgray', alpha=0.5))

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return None
        else:
            return fig

    def generate_report(self, output_dir):
        """
        Generate visual report for all measurements.

        Args:
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, measurement in enumerate(self.measurements):
            # Create individual visualizations
            self.visualize_measurements(
                measurement,
                save_path=os.path.join(output_dir,
                                       f"patient_{measurement['patient_id']}_{measurement['timepoint']}_measurements.png")
            )

            # Create separate visualizations for lateral pillar and EQ
            self.visualize_lateral_pillar(
                measurement, 'affected',
                save_path=os.path.join(output_dir,
                                       f"patient_{measurement['patient_id']}_{measurement['timepoint']}_affected_lateral.png")
            )

            self.visualize_lateral_pillar(
                measurement, 'unaffected',
                save_path=os.path.join(output_dir,
                                       f"patient_{measurement['patient_id']}_{measurement['timepoint']}_unaffected_lateral.png")
            )

            self.visualize_epiphyseal_quotient(
                measurement, 'affected',
                save_path=os.path.join(output_dir,
                                       f"patient_{measurement['patient_id']}_{measurement['timepoint']}_affected_eq.png")
            )

            self.visualize_epiphyseal_quotient(
                measurement, 'unaffected',
                save_path=os.path.join(output_dir,
                                       f"patient_{measurement['patient_id']}_{measurement['timepoint']}_unaffected_eq.png")
            )

        # Save CSV report
        self.to_csv(os.path.join(output_dir, "perthes_measurements.csv"))
        print(f"Generated report with {len(self.measurements)} cases in {output_dir}")