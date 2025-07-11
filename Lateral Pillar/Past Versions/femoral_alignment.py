import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from skimage.measure import label, regionprops
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist


class FemoralMaskAligner:
    def __init__(self, coco_path, image_dir, output_dir):
        self.coco_path = coco_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load COCO data
        with open(coco_path) as f:
            self.coco_data = json.load(f)

        # Create mappings
        self._create_mappings()
        self.head_cat_id = next(c['id'] for c in self.coco_data['categories']
                                if c['name'].lower() == 'head')

    def _create_mappings(self):
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {img_id: [] for img_id in self.image_id_to_info.keys()}

        for ann in self.coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def _parse_filename(self, filename):
        parts = os.path.splitext(filename)[0].split('_')
        return {
            'patient': parts[1],
            'view': parts[2],
            'timepoint': parts[3],
            'laterality': parts[5]
        }

    def _get_head_mask(self, img_id):
        mask = np.zeros((self.image_id_to_info[img_id]['height'],
                         self.image_id_to_info[img_id]['width']), dtype=np.uint8)

        for ann in self.annotations.get(img_id, []):
            if ann['category_id'] == self.head_cat_id:
                seg = np.array(ann['segmentation'][0]).reshape(-1, 2)
                cv2.fillPoly(mask, [seg.astype(int)], 1)
        return mask

    def _flip_mask(self, mask):
        return np.fliplr(mask).copy()

    def _dice_score(self, mask1, mask2):
        """Custom Dice score implementation"""
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        intersection = np.logical_and(mask1, mask2)
        return (2.0 * intersection.sum()) / (mask1.sum() + mask2.sum() + 1e-7)

    def _centroid_alignment(self, source, target):
        src_props = regionprops(label(source))[0]
        tgt_props = regionprops(label(target))[0]
        return np.array([
            [1, 0, tgt_props.centroid[1] - src_props.centroid[1]],
            [0, 1, tgt_props.centroid[0] - src_props.centroid[0]],
            [0, 0, 1]
        ])

    def _corner_alignment(self, source, target):
        src_points = np.argwhere(source)
        tgt_points = np.argwhere(target)

        # Find rightmost-lowermost point
        src_ref = src_points[np.lexsort((src_points[:, 1], -src_points[:, 0]))[0]]
        tgt_ref = tgt_points[np.lexsort((tgt_points[:, 1], -tgt_points[:, 0]))[0]]

        return np.array([
            [1, 0, tgt_ref[1] - src_ref[1]],
            [0, 1, tgt_ref[0] - src_ref[0]],
            [0, 0, 1]
        ])

    def _principal_axis_alignment(self, source, target):
        """Align masks based on their principal axes"""
        src_props = regionprops(label(source))[0]
        tgt_props = regionprops(label(target))[0]

        # Get orientation angles
        src_angle = src_props.orientation
        tgt_angle = tgt_props.orientation

        # Calculate rotation angle difference
        rot_angle = tgt_angle - src_angle

        # Create rotation matrix
        rot_mat = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle), 0],
            [np.sin(rot_angle), np.cos(rot_angle), 0],
            [0, 0, 1]
        ])

        # Translate to centroids
        translate_to_src = np.array([
            [1, 0, -src_props.centroid[1]],
            [0, 1, -src_props.centroid[0]],
            [0, 0, 1]
        ])

        translate_to_tgt = np.array([
            [1, 0, tgt_props.centroid[1]],
            [0, 1, tgt_props.centroid[0]],
            [0, 0, 1]
        ])

        # Combine transformations
        return translate_to_tgt @ rot_mat @ translate_to_src

    def _procrustes_alignment(self, source, target):
        """Align using Procrustes analysis"""
        src_points = np.argwhere(source)
        tgt_points = np.argwhere(target)

        # Sample points if too many
        if len(src_points) > 1000:
            src_points = src_points[np.random.choice(len(src_points), 1000, replace=False)]
        if len(tgt_points) > 1000:
            tgt_points = tgt_points[np.random.choice(len(tgt_points), 1000, replace=False)]

        # Center points
        src_center = np.mean(src_points, axis=0)
        tgt_center = np.mean(tgt_points, axis=0)
        src_points_centered = src_points - src_center
        tgt_points_centered = tgt_points - tgt_center

        # Compute rotation matrix
        H = src_points_centered.T @ tgt_points_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Build transformation matrix
        T = tgt_center - R @ src_center
        return np.array([
            [R[0, 0], R[0, 1], T[1]],
            [R[1, 0], R[1, 1], T[0]],
            [0, 0, 1]
        ])

    def _chamfer_alignment(self, source, target, iterations=20):
        """Align using Chamfer distance minimization"""
        # Create distance transform of target
        target_dt = cv2.distanceTransform(
            target.astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        )

        # Get source points
        src_points = np.argwhere(source)

        # Initial transformation (centroid alignment)
        M = self._centroid_alignment(source, target)
        best_score = float('inf')
        best_M = M

        # Try random perturbations
        for _ in range(iterations):
            # Perturb transformation
            perturb = np.eye(3)
            perturb[0, 2] = np.random.uniform(-10, 10)
            perturb[1, 2] = np.random.uniform(-10, 10)
            perturb[0, 0] = 1 + np.random.uniform(-0.1, 0.1)
            perturb[1, 1] = 1 + np.random.uniform(-0.1, 0.1)
            perturb[0, 1] = np.random.uniform(-0.05, 0.05)
            perturb[1, 0] = np.random.uniform(-0.05, 0.05)

            test_M = perturb @ M

            # Transform points
            transformed_points = (test_M[:2, :2] @ src_points.T + test_M[:2, 2:3]).T

            # Calculate chamfer distance
            distances = cdist(transformed_points, np.argwhere(target))
            min_distances = np.min(distances, axis=1)
            score = np.mean(min_distances)

            # Update best transformation
            if score < best_score:
                best_score = score
                best_M = test_M

        return best_M

    def _rigid_alignment(self, source, target):
        # Feature-based alignment using ORB
        src_pts, tgt_pts = self._find_keypoints(source, target)

        if len(src_pts) < 4:
            return self._centroid_alignment(source, target)

        M = cv2.estimateAffinePartial2D(src_pts, tgt_pts)[0]
        if M is None:
            return self._centroid_alignment(source, target)

        # Convert to 3x3 homogeneous matrix
        M_hom = np.vstack([M, [0, 0, 1]])
        return M_hom

    def _find_keypoints(self, source, target):
        # Convert to uint8 for OpenCV
        source_uint8 = (source * 255).astype(np.uint8)
        target_uint8 = (target * 255).astype(np.uint8)

        # Find keypoints and descriptors
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(source_uint8, None)
        kp2, des2 = orb.detectAndCompute(target_uint8, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return [], []

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:10]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        tgt_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        return src_pts, tgt_pts

    def align_and_evaluate(self):
        results = []

        # Group images by patient, view, and timepoint
        groups = {}
        for img_info in self.coco_data['images']:
            meta = self._parse_filename(img_info['file_name'])
            key = (meta['patient'], meta['view'], meta['timepoint'])
            groups.setdefault(key, {})[meta['laterality']] = img_info['id']

        # Process each group
        for (patient, view, timepoint), ids in groups.items():
            if 'L' not in ids or 'R' not in ids:
                continue

            # Get masks
            left_mask = self._get_head_mask(ids['L'])
            right_mask = self._get_head_mask(ids['R'])

            # Flip healthy mask (left)
            healthy_flipped = self._flip_mask(left_mask)

            # Alignment methods
            methods = {
                'Centroid': self._centroid_alignment,
                'Corner': self._corner_alignment,
                'PrincipalAxis': self._principal_axis_alignment,
                'Procrustes': self._procrustes_alignment,
                'Chamfer': lambda s, t: self._chamfer_alignment(s, t),
                'Rigid': self._rigid_alignment
            }

            # Process each method
            for method_name, align_func in methods.items():
                try:
                    # Get transformation matrix (3x3 homogeneous)
                    M = align_func(healthy_flipped, right_mask)

                    # Apply transformation
                    aligned = affine_transform(
                        healthy_flipped,
                        M[:2, :2],  # rotation/scale
                        offset=M[:2, 2],  # translation
                        output_shape=right_mask.shape
                    ) > 0.5

                    # Calculate Dice score
                    dice = self._dice_score(right_mask, aligned)

                    # Visualize results
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f"{method_name} Alignment (Dice: {dice:.3f})")

                    ax[0].imshow(left_mask, cmap='gray')
                    ax[0].set_title('Original Healthy (Left)')

                    ax[1].imshow(right_mask, cmap='gray')
                    ax[1].set_title('Affected (Right)')

                    # Create color overlay
                    overlay = np.zeros((*right_mask.shape, 3), dtype=np.uint8)
                    overlay[..., 0] = right_mask.astype(np.uint8) * 255  # Red for affected
                    overlay[..., 1] = aligned.astype(np.uint8) * 255  # Green for healthy

                    ax[2].imshow(overlay)
                    ax[2].set_title('Overlay (Green: Healthy, Red: Affected)')

                    # Save results
                    plt.tight_layout()
                    fname = f"Patient_{patient}_{view}_{timepoint}_{method_name}.png"
                    plt.savefig(os.path.join(self.output_dir, fname))
                    plt.close()

                    results.append({
                        'patient': patient,
                        'view': view,
                        'timepoint': timepoint,
                        'method': method_name,
                        'dice': dice
                    })
                except Exception as e:
                    print(f"Error with {method_name} for {patient} {view} {timepoint}: {str(e)}")
                    results.append({
                        'patient': patient,
                        'view': view,
                        'timepoint': timepoint,
                        'method': method_name,
                        'dice': -1,
                        'error': str(e)
                    })

        return results


# Usage example
if __name__ == "__main__":
    aligner = FemoralMaskAligner(
        coco_path=r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json',
        image_dir=r'C:\Users\SR207348\Downloads\ipsg102\ipsg102',
        output_dir=r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_lateral_pillar"
    )
    results = aligner.align_and_evaluate()
    print("Alignment Results:")
    for res in results:
        if 'error' in res:
            print(f"{res['patient']} {res['view']} {res['timepoint']}: "
                  f"{res['method']} failed - {res['error']}")
        else:
            print(f"{res['patient']} {res['view']} {res['timepoint']}: "
                  f"{res['method']} Dice = {res['dice']:.4f}")


#coco_path=r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json',
#image_dir=r'C:\Users\SR207348\Downloads\ipsg102\ipsg102',
#output_dir=r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_lateral_pillar"