import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import cv2


class FemoralHeadReconstructor:
    def __init__(self, coco_json_path):
        """
        Initialize the reconstructor with COCO format mask data
        """
        self.coco_data = self.load_coco_json(coco_json_path)
        self.mask = None
        self.rotated_mask = None
        self.rotation_angle = 0
        self.original_volume = 0
        self.epiphysis_mask = None
        self.reconstructed_mask = None
        self.final_mask = None

    def load_coco_json(self, json_path):
        """Load COCO format JSON file"""
        with open(json_path, 'r') as f:
            return json.load(f)

    def coco_to_mask(self, image_id=None, annotation_id=None):
        """
        Convert COCO annotation to binary mask
        If multiple annotations exist, specify which one to use
        """
        # Get image info
        if image_id is None:
            image_id = self.coco_data['images'][0]['id']

        image_info = next(img for img in self.coco_data['images'] if img['id'] == image_id)
        height, width = image_info['height'], image_info['width']

        # Get annotation
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        if annotation_id is not None:
            annotation = next(ann for ann in annotations if ann['id'] == annotation_id)
        else:
            annotation = annotations[0]  # Use first annotation if not specified

        # Create mask from segmentation
        mask = np.zeros((height, width), dtype=np.uint8)

        if 'segmentation' in annotation:
            for segmentation in annotation['segmentation']:
                # Convert segmentation to polygon points
                poly = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

        self.mask = mask.astype(bool)
        self.original_volume = np.sum(self.mask)
        return self.mask

    def find_major_axis_and_rotate(self):
        """
        Find the major axis of the femoral head and rotate so it's horizontal
        """
        # Get coordinates of mask pixels
        coords = np.column_stack(np.where(self.mask))

        if len(coords) < 2:
            print("Warning: Not enough points for PCA analysis")
            self.rotated_mask = self.mask.copy()
            self.rotation_angle = 0
            return self.rotated_mask, 0

        # Use PCA to find the major axis
        pca = PCA(n_components=2)
        pca.fit(coords)

        # Get the first principal component (major axis direction)
        major_axis = pca.components_[0]

        # Calculate angle to rotate major axis to horizontal
        angle = np.arctan2(major_axis[0], major_axis[1])
        self.rotation_angle = -np.degrees(angle)

        print(f"Rotating mask by {self.rotation_angle:.2f} degrees to align major axis horizontally")

        # Rotate the mask
        center = (self.mask.shape[1] // 2, self.mask.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)

        self.rotated_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            rotation_matrix,
            (self.mask.shape[1], self.mask.shape[0])
        ).astype(bool)

        return self.rotated_mask, self.rotation_angle

    def find_femoral_head_center(self, mask=None):
        """
        Find the center of the femoral head using centroid and geometric analysis
        """
        if mask is None:
            mask = self.rotated_mask

        # Get coordinates of mask pixels
        coords = np.column_stack(np.where(mask))

        # Calculate centroid
        centroid = np.mean(coords, axis=0)

        # Find the topmost point (assuming head is at top after rotation)
        top_points = coords[coords[:, 0] == np.min(coords[:, 0])]
        top_center = np.mean(top_points, axis=0)

        # Estimate center as weighted average of centroid and top center
        center = 0.7 * centroid + 0.3 * top_center

        return center.astype(int)

    def segment_epiphysis_and_head(self, split_ratio=0.6):
        """
        Segment the rotated mask into epiphysis (bottom) and head (top) portions
        split_ratio: fraction of height to consider as epiphysis
        """
        coords = np.column_stack(np.where(self.rotated_mask))
        min_row, max_row = np.min(coords[:, 0]), np.max(coords[:, 0])

        # Calculate split point
        split_row = min_row + (max_row - min_row) * split_ratio

        # Create epiphysis mask (bottom part)
        self.epiphysis_mask = self.rotated_mask.copy()
        self.epiphysis_mask[:int(split_row), :] = False

        # Head mask (top part) - will be replaced
        head_mask = self.rotated_mask.copy()
        head_mask[int(split_row):, :] = False

        return self.epiphysis_mask, head_mask, int(split_row)

    def get_boundary_points(self, split_row):
        """
        Get the boundary points where the epiphysis meets the split line
        """
        # Find points on the split row that are part of the epiphysis
        boundary_points = []

        # Check the split row for epiphysis pixels
        split_line = self.epiphysis_mask[split_row, :]
        boundary_cols = np.where(split_line)[0]

        if len(boundary_cols) > 0:
            # Get leftmost and rightmost points
            left_point = (split_row, boundary_cols[0])
            right_point = (split_row, boundary_cols[-1])
            boundary_points = [left_point, right_point]

        return boundary_points

    def create_smooth_curved_head(self, center, radius, split_row, boundary_points):
        """
        Create a smooth curved head that connects seamlessly with the epiphysis
        """
        mask_shape = self.rotated_mask.shape
        y_coords, x_coords = np.ogrid[:mask_shape[0], :mask_shape[1]]

        # Create the basic spherical/elliptical shape
        distances = np.sqrt((x_coords - center[1]) ** 2 + (y_coords - center[0]) ** 2)
        sphere_mask = distances <= radius

        # Only keep the part above the split line
        sphere_mask[split_row:, :] = False

        # If we have boundary points, ensure smooth connection
        if len(boundary_points) >= 2:
            left_point, right_point = boundary_points[0], boundary_points[1]

            # Create a smooth transition by modifying the sphere near the boundary
            for row in range(max(0, split_row - 10), split_row + 1):
                # Get the width of the epiphysis at this row
                if row < mask_shape[0]:
                    epiphysis_row = self.epiphysis_mask[row, :]
                    if np.any(epiphysis_row):
                        epiphysis_cols = np.where(epiphysis_row)[0]
                        if len(epiphysis_cols) > 0:
                            left_col = epiphysis_cols[0]
                            right_col = epiphysis_cols[-1]

                            # Create smooth transition by filling between these points
                            # with a curved profile
                            center_col = (left_col + right_col) // 2
                            width = right_col - left_col

                            # Create a parabolic profile for smooth connection
                            for col in range(left_col, right_col + 1):
                                # Distance from center of this row
                                dist_from_center = abs(col - center_col)
                                normalized_dist = dist_from_center / (width / 2) if width > 0 else 0

                                # Parabolic curve (inverted)
                                if normalized_dist <= 1:
                                    sphere_mask[row, col] = True

        return sphere_mask

    def find_optimal_radius(self, center, split_row, target_volume):
        def find_optimal_radius(self, center, split_row, boundary_points, target_volume):
            """
            Find the optimal radius for the curved head to match target volume
            """

            def volume_difference(radius):
                cap_mask = self.create_smooth_curved_head(center, radius, split_row, boundary_points)
                current_volume = np.sum(cap_mask)
                return abs(current_volume - target_volume)

            # Search for optimal radius with bounds
            try:
                result = minimize_scalar(volume_difference, bounds=(5, 100), method='bounded')
                return result.x
            except Exception as e:
                print(f"Optimization failed: {e}")
                return 25.0  # fallback radius

    def rotate_back_to_original(self, mask):
        """
        Rotate the reconstructed mask back to the original orientation
        """
        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -self.rotation_angle, 1.0)

        rotated_back = cv2.warpAffine(
            mask.astype(np.uint8),
            rotation_matrix,
            (mask.shape[1], mask.shape[0])
        ).astype(bool)

        return rotated_back

    def reconstruct_femoral_head(self, split_ratio=0.6):
        """
        Main reconstruction function with rotation alignment
        """
        print("Starting femoral head reconstruction...")

        # Step 1: Rotate the mask so major axis is horizontal
        self.find_major_axis_and_rotate()

        # Step 2: Segment into epiphysis and head
        epiphysis_mask, head_mask, split_row = self.segment_epiphysis_and_head(split_ratio)

        # Step 3: Get boundary points for smooth connection
        boundary_points = self.get_boundary_points(split_row)

        # Calculate volumes
        epiphysis_volume = np.sum(epiphysis_mask)
        head_volume = np.sum(head_mask)

        print(f"Original total volume: {self.original_volume}")
        print(f"Epiphysis volume: {epiphysis_volume}")
        print(f"Original head volume: {head_volume}")
        print(f"Boundary points found: {len(boundary_points)}")

        # Step 4: Find center for reconstruction
        center = self.find_femoral_head_center()

        # Step 5: Find optimal radius for equal volume
        target_head_volume = epiphysis_volume  # Make head volume equal to epiphysis

        # Define volume matching function
        def volume_difference(radius):
            cap_mask = self.create_smooth_curved_head(center, radius, split_row, boundary_points)
            current_volume = np.sum(cap_mask)
            return abs(current_volume - target_head_volume)

        # Find optimal radius
        result = minimize_scalar(volume_difference, bounds=(5, 100), method='bounded')
        optimal_radius = result.x

        print(f"Optimal radius for reconstruction: {optimal_radius:.2f}")

        # Step 6: Create the reconstructed curved head
        reconstructed_head = self.create_smooth_curved_head(center, optimal_radius, split_row, boundary_points)

        # Step 7: Combine epiphysis with reconstructed head
        self.reconstructed_mask = epiphysis_mask | reconstructed_head

        # Step 8: Rotate back to original orientation
        self.final_mask = self.rotate_back_to_original(self.reconstructed_mask)

        # Verify volumes
        new_head_volume = np.sum(reconstructed_head)
        new_total_volume = np.sum(self.reconstructed_mask)

        print(f"New head volume: {new_head_volume}")
        print(f"New total volume: {new_total_volume}")
        print(f"Volume ratio (head/epiphysis): {new_head_volume / epiphysis_volume:.3f}")

        return self.final_mask

    def visualize_reconstruction(self):
        def visualize_reconstruction(self):
            """
            Visualize the original and reconstructed masks
            """
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Top row: Rotated view (working orientation)
            axes[0, 0].imshow(self.mask, cmap='gray')
            axes[0, 0].set_title('Original Mask')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(self.rotated_mask, cmap='gray')
            axes[0, 1].set_title('Rotated (Major Axis Horizontal)')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(self.reconstructed_mask, cmap='gray')
            axes[0, 2].set_title('Reconstructed (Rotated View)')
            axes[0, 2].axis('off')

            # Bottom row: Final results
            axes[1, 0].imshow(self.mask, cmap='gray')
            axes[1, 0].set_title('Original Deformed Head')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(self.rotate_back_to_original(self.epiphysis_mask), cmap='gray')
            axes[1, 1].set_title('Epiphysis (Preserved)')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(self.final_mask, cmap='gray')
            axes[1, 2].set_title('Final Reconstructed Head')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.show()

    def visualize_differences(self):
        """
        Visualize the differences between original and reconstructed masks
        """
        if self.final_mask is None:
            print("No reconstructed mask available. Run reconstruct_femoral_head() first.")
            return

        # Create difference maps
        added_regions = self.final_mask & ~self.mask  # New regions added
        removed_regions = self.mask & ~self.final_mask  # Original regions removed
        unchanged_regions = self.mask & self.final_mask  # Regions that stayed the same

        # Create colored overlay
        overlay = np.zeros((*self.mask.shape, 3))  # RGB image
        overlay[unchanged_regions] = [0.5, 0.5, 0.5]  # Gray for unchanged
        overlay[added_regions] = [0, 1, 0]  # Green for added
        overlay[removed_regions] = [1, 0, 0]  # Red for removed

        # Calculate statistics
        added_volume = np.sum(added_regions)
        removed_volume = np.sum(removed_regions)
        unchanged_volume = np.sum(unchanged_regions)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Top row: Individual difference maps
        axes[0, 0].imshow(added_regions, cmap='Greens')
        axes[0, 0].set_title(f'Added Regions\n(Volume: {added_volume} pixels)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(removed_regions, cmap='Reds')
        axes[0, 1].set_title(f'Removed Regions\n(Volume: {removed_volume} pixels)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(unchanged_regions, cmap='gray')
        axes[0, 2].set_title(f'Unchanged Regions\n(Volume: {unchanged_volume} pixels)')
        axes[0, 2].axis('off')

        # Bottom row: Comparisons and overlays
        axes[1, 0].imshow(self.mask, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(self.final_mask, cmap='Blues', alpha=0.3)
        axes[1, 0].set_title('Original (Gray) + Reconstructed (Blue)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Difference Map\n(Red: Removed, Green: Added, Gray: Unchanged)')
        axes[1, 1].axis('off')

        # Side by side comparison
        comparison = np.zeros((self.mask.shape[0], self.mask.shape[1] * 2))
        comparison[:, :self.mask.shape[1]] = self.mask
        comparison[:, self.mask.shape[1]:] = self.final_mask
        axes[1, 2].imshow(comparison, cmap='gray')
        axes[1, 2].axvline(x=self.mask.shape[1], color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_title('Side by Side\n(Left: Original, Right: Reconstructed)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        # Print statistics
        print("\n" + "=" * 50)
        print("RECONSTRUCTION STATISTICS")
        print("=" * 50)
        print(f"Original volume: {np.sum(self.mask)} pixels")
        print(f"Reconstructed volume: {np.sum(self.final_mask)} pixels")
        print(f"Volume change: {np.sum(self.final_mask) - np.sum(self.mask):+d} pixels")
        print(f"Added regions: {added_volume} pixels")
        print(f"Removed regions: {removed_volume} pixels")
        print(f"Unchanged regions: {unchanged_volume} pixels")
        print(f"Overlap percentage: {(unchanged_volume / np.sum(self.mask)) * 100:.1f}%")

    def visualize_cross_sections(self, num_sections=5):
        """
        Visualize cross-sections through the femoral head to show internal differences
        """
        if self.final_mask is None:
            print("No reconstructed mask available. Run reconstruct_femoral_head() first.")
            return

        # Get coordinates for cross-sections
        coords = np.column_stack(np.where(self.mask))
        min_row, max_row = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_col, max_col = np.min(coords[:, 1]), np.max(coords[:, 1])

        # Create cross-section positions
        section_rows = np.linspace(min_row, max_row, num_sections).astype(int)
        section_cols = np.linspace(min_col, max_col, num_sections).astype(int)

        fig, axes = plt.subplots(2, num_sections, figsize=(15, 6))

        # Horizontal cross-sections
        for i, row in enumerate(section_rows):
            original_section = self.mask[row, :]
            reconstructed_section = self.final_mask[row, :]

            x = np.arange(len(original_section))
            axes[0, i].fill_between(x, 0, original_section, alpha=0.7, color='red', label='Original')
            axes[0, i].fill_between(x, 0, reconstructed_section, alpha=0.5, color='blue', label='Reconstructed')
            axes[0, i].set_title(f'Horizontal Section {i + 1}\n(Row {row})')
            axes[0, i].set_ylim(0, 1.2)
            if i == 0:
                axes[0, i].legend()

        # Vertical cross-sections
        for i, col in enumerate(section_cols):
            original_section = self.mask[:, col]
            reconstructed_section = self.final_mask[:, col]

            y = np.arange(len(original_section))
            axes[1, i].fill_betweenx(y, 0, original_section, alpha=0.7, color='red', label='Original')
            axes[1, i].fill_betweenx(y, 0, reconstructed_section, alpha=0.5, color='blue', label='Reconstructed')
            axes[1, i].set_title(f'Vertical Section {i + 1}\n(Col {col})')
            axes[1, i].set_xlim(0, 1.2)
            axes[1, i].invert_yaxis()  # Invert y-axis to match image orientation
            if i == 0:
                axes[1, i].legend()

        plt.tight_layout()
        plt.show()

    def create_3d_comparison(self):
        """
        Create a 3D visualization comparing original and reconstructed masks
        """
        if self.final_mask is None:
            print("No reconstructed mask available. Run reconstruct_femoral_head() first.")
            return

        try:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(15, 5))

            # Original mask 3D
            ax1 = fig.add_subplot(131, projection='3d')
            coords_orig = np.column_stack(np.where(self.mask))
            if len(coords_orig) > 0:
                ax1.scatter(coords_orig[:, 1], coords_orig[:, 0],
                            np.zeros_like(coords_orig[:, 0]),
                            c='red', alpha=0.6, s=1)
            ax1.set_title('Original Mask')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')

            # Reconstructed mask 3D
            ax2 = fig.add_subplot(132, projection='3d')
            coords_recon = np.column_stack(np.where(self.final_mask))
            if len(coords_recon) > 0:
                ax2.scatter(coords_recon[:, 1], coords_recon[:, 0],
                            np.zeros_like(coords_recon[:, 0]),
                            c='blue', alpha=0.6, s=1)
            ax2.set_title('Reconstructed Mask')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

            # Overlay comparison
            ax3 = fig.add_subplot(133, projection='3d')
            if len(coords_orig) > 0:
                ax3.scatter(coords_orig[:, 1], coords_orig[:, 0],
                            np.zeros_like(coords_orig[:, 0]),
                            c='red', alpha=0.4, s=1, label='Original')
            if len(coords_recon) > 0:
                ax3.scatter(coords_recon[:, 1], coords_recon[:, 0],
                            np.ones_like(coords_recon[:, 0]) * 0.1,
                            c='blue', alpha=0.4, s=1, label='Reconstructed')
            ax3.set_title('Overlay Comparison')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("3D visualization requires matplotlib with 3D support")
        except Exception as e:
            print(f"3D visualization failed: {e}")

    def save_reconstructed_mask(self, output_path):
        def save_reconstructed_mask(self, output_path):
            """
            Save the reconstructed mask as a new COCO format JSON
            """
            if self.final_mask is None:
                raise ValueError("No reconstructed mask available. Run reconstruct_femoral_head() first.")

            # Convert mask to COCO segmentation format
            contours, _ = cv2.findContours(
                self.final_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Create new COCO data structure
            new_coco_data = {
                "images": self.coco_data["images"].copy(),
                "annotations": [],
                "categories": self.coco_data.get("categories", [])
            }

            # Convert contours to segmentation format
            for i, contour in enumerate(contours):
                if len(contour) > 2:  # Valid contour
                    segmentation = contour.flatten().tolist()

                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)

                    annotation = {
                        "id": i + 1,
                        "image_id": self.coco_data["images"][0]["id"],
                        "category_id": self.coco_data["annotations"][0].get("category_id", 1),
                        "segmentation": [segmentation],
                        "area": float(area),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "iscrowd": 0
                    }

                    new_coco_data["annotations"].append(annotation)

            # Save to file
            with open(output_path, 'w') as f:
                json.dump(new_coco_data, f, indent=2)

            print(f"Reconstructed mask saved to: {output_path}")


# Usage example
def main():
    # Initialize reconstructor
    reconstructor = FemoralHeadReconstructor('path/to/your/coco.json')

    # Convert COCO to mask
    mask = reconstructor.coco_to_mask()

    # Perform reconstruction
    reconstructed_mask = reconstructor.reconstruct_femoral_head(split_ratio=0.6)

    # Visualize results
    reconstructor.visualize_reconstruction()

    # Visualize differences
    reconstructor.visualize_differences()

    # Visualize cross-sections
    reconstructor.visualize_cross_sections()

    # Create 3D comparison
    reconstructor.create_3d_comparison()

    # Save reconstructed mask
    reconstructor.save_reconstructed_mask('reconstructed_femoral_head.json')


if __name__ == "__main__":
    # Replace with your actual file path
    coco_file_path = r"C:\Users\SR207348\PyCharmMiscProject\.venv\HIPMETRICS\Annotations\output ipsg106.json"

    reconstructor = FemoralHeadReconstructor(coco_file_path)
    mask = reconstructor.coco_to_mask()
    reconstructed_mask = reconstructor.reconstruct_femoral_head()

    # Basic visualization
    reconstructor.visualize_reconstruction()

    # Detailed difference analysis
    reconstructor.visualize_differences()

    # Cross-section analysis
    reconstructor.visualize_cross_sections()

    # 3D comparison
    reconstructor.create_3d_comparison()

    # Save result
    reconstructor.save_reconstructed_mask('reconstructed_femoral_head.json')