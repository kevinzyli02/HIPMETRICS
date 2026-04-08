import numpy as np
import cv2
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree


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
        # Extract lateral pillar max average (index 0 in both lists)
        aff_lat_avg = aff_heights[1]
        unaff_lat_avg = unaff_heights[1]

        # Handle edge cases
        if unaff_lat_avg == 0:
            return "Undefined (reference height=0)"

        # Calculate height ratio
        height_ratio = aff_lat_avg / unaff_lat_avg

        # Determine classification
        if height_ratio >= 0.95:
            return "A"
        elif height_ratio >= 0.5:
            return "B"
        else:
            return "C"

    def _get_lateral_boundaries(self, mask, laterality):
        """Calculate lateral pillar boundaries for visualization"""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x

        # Define pillar boundaries
        left_bound = min_x
        middle_start = min_x + 0.25 * width
        middle_end = min_x + 0.75 * width
        right_bound = max_x

        # Determine lateral pillar boundaries based on laterality
        if laterality == 'R':
            return (middle_end, right_bound)  # Right hip: lateral pillar is rightmost
        else:  # 'L'
            return (left_bound, middle_start)  # Left hip: lateral pillar is leftmost


    def calculate(self):
        """Calculate pillar measurements and determine Herring classification"""
        try:
            # Measure for affected and unaffected masks
            aff_heights = self._calculate_pillar_heights(
                self.analyzer.rotated_aff_mask, self.data['affected_laterality']
            )
            unaff_heights = self._calculate_pillar_heights(
                self.analyzer.rotated_unaff_mask, self.data['affected_laterality'] #since it is already flipped l/r we want to use the same laterality as the affected side
            )

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

    @staticmethod
    def _extract_contour_points(mask, subsample_ratio=0.25):
        """Extract evenly-spaced contour points from a binary mask."""
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea).squeeze()
        if contour.ndim == 1:
            contour = contour.reshape(1, 2)
        n = max(3, int(len(contour) * subsample_ratio))
        step = max(1, len(contour) // n)
        return contour[::step].astype(float)

    @staticmethod
    def icp_align(source_mask, target_mask, subsample_ratio=0.25, max_iter=50,
                  tol=1e-4, max_rotation_deg=20):
        """
        Align source_mask to target_mask using ICP on contour point clouds.

        Uses 25% random point subsampling (per the paper) and caps cumulative rotation
        at max_rotation_deg to prevent runaway alignment on already-oriented masks.

        Returns (R, t, error): 2x2 rotation matrix, 2D translation, mean registration error.
        """
        source_pts = DIMeasurement._extract_contour_points(source_mask, subsample_ratio)
        target_pts = DIMeasurement._extract_contour_points(target_mask, subsample_ratio)

        if source_pts is None or target_pts is None or len(source_pts) < 3 or len(target_pts) < 3:
            print("ICP: insufficient contour points, returning identity transform")
            return np.eye(2), np.zeros(2), float('nan')

        tree = cKDTree(target_pts)
        R_total = np.eye(2)
        t_total = np.zeros(2)
        src = source_pts.copy()
        prev_error = float('inf')

        for _ in range(max_iter):
            distances, indices = tree.query(src)
            matched_target = target_pts[indices]

            src_centroid = src.mean(axis=0)
            tgt_centroid = matched_target.mean(axis=0)

            H_mat = (src - src_centroid).T @ (matched_target - tgt_centroid)
            U, _, Vt = np.linalg.svd(H_mat)
            R = Vt.T @ U.T

            # Ensure proper rotation (det = +1, not a reflection)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Clamp cumulative rotation to max_rotation_deg
            candidate_R = R @ R_total
            total_angle = np.degrees(np.arctan2(candidate_R[1, 0], candidate_R[0, 0]))
            if abs(total_angle) > max_rotation_deg:
                R = np.eye(2)

            t = tgt_centroid - R @ src_centroid
            src = (R @ src.T).T + t
            R_total = R @ R_total
            t_total = R @ t_total + t

            error = distances.mean()
            if abs(prev_error - error) < tol:
                break
            prev_error = error

        final_distances, _ = tree.query(src)
        return R_total, t_total, float(final_distances.mean())

    @staticmethod
    def _apply_rigid_transform_to_mask(mask, R, t):
        """
        Apply a 2D rigid transform (R, t) to a binary mask image.
        Pixels at source position p are mapped to R @ p + t in the output.
        """
        H, W = mask.shape
        M = np.hstack([R, t.reshape(2, 1)]).astype(np.float64)
        aligned = cv2.warpAffine(mask.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        return aligned.astype(bool)

    @staticmethod
    def extract_medial_column_landmarks(head_mask, neck_mask, laterality):
        """
        Extract the three medial-column anatomical landmarks:
          pt1 – topmost medial point of the femoral head (epiphysis medial edge)
          pt2 – bottommost medial point of the femoral head (growth-plate junction)
          pt3 – most medial point of the femoral neck (metaphysis medial edge)

        For a right hip (R) the medial side is the left/min-x side; for left (L) it
        is the right/max-x side.  "Medial" here means the inner 25 % of the head's
        horizontal extent.

        Returns a (3, 2) float array [[x1,y1],[x2,y2],[x3,y3]], or None if either
        mask is absent or too sparse to locate reliable landmarks.
        """
        if head_mask is None or not head_mask.any():
            return None
        if neck_mask is None or not neck_mask.any():
            return None

        head_ys, head_xs = np.where(head_mask)
        x_span = int(head_xs.max()) - int(head_xs.min())
        if x_span < 4:
            return None

        medial_width = max(1, int(x_span * 0.25))
        if laterality == 'R':
            mask_col = head_xs <= head_xs.min() + medial_width
            pick_x   = np.min
        else:
            mask_col = head_xs >= head_xs.max() - medial_width
            pick_x   = np.max

        med_ys = head_ys[mask_col]
        med_xs = head_xs[mask_col]
        if med_ys.size == 0:
            return None

        pt1_y = int(med_ys.min())
        pt2_y = int(med_ys.max())
        pt1_x = int(pick_x(med_xs[med_ys == pt1_y]))
        pt2_x = int(pick_x(med_xs[med_ys == pt2_y]))

        # pt3: most-medial pixel of the neck mask
        nk_ys, nk_xs = np.where(neck_mask)
        if laterality == 'R':
            pt3_x = int(nk_xs.min())
            col_ys = nk_ys[nk_xs == pt3_x]
        else:
            pt3_x = int(nk_xs.max())
            col_ys = nk_ys[nk_xs == pt3_x]

        pt3_y = int(np.mean(col_ys)) if col_ys.size > 0 else int(nk_ys.mean())

        return np.array([[pt1_x, pt1_y],
                         [pt2_x, pt2_y],
                         [pt3_x, pt3_y]], dtype=float)

    @staticmethod
    def compute_rigid_from_landmarks(src_pts, tgt_pts):
        """
        Compute the rigid transform (R, t) that maps src_pts -> tgt_pts using SVD.
        src_pts, tgt_pts: (N, 2) float arrays (N >= 2).
        Returns R (2×2), t (2,).
        """
        src = np.asarray(src_pts, dtype=float)
        tgt = np.asarray(tgt_pts, dtype=float)
        src_c = src.mean(axis=0)
        tgt_c = tgt.mean(axis=0)
        H = (src - src_c).T @ (tgt - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_c - R @ src_c
        return R, t

    @staticmethod
    def icp_align_weighted(source_pts, target_pts, landmark_pts=None,
                           landmark_weight=5.0, max_iter=50, tol=1e-4,
                           max_rotation_deg=20):
        """
        ICP on pre-extracted point arrays with optional landmark upweighting.

        source_pts, target_pts : (N, 2) and (M, 2) float arrays (contour points)
        landmark_pts           : (K, 2) float array of high-importance source points
                                 that are appended with weight `landmark_weight`
                                 (i.e. repeated landmark_weight times).
        Returns (R_total, t_total, mean_error).
        """
        if (source_pts is None or target_pts is None
                or len(source_pts) < 3 or len(target_pts) < 3):
            return np.eye(2), np.zeros(2), float('nan')

        # Augment source with upweighted landmarks
        if landmark_pts is not None and len(landmark_pts) > 0:
            repeats = max(1, int(round(landmark_weight)))
            aug = np.tile(landmark_pts, (repeats, 1))
            src = np.vstack([source_pts, aug]).astype(float)
        else:
            src = source_pts.astype(float).copy()

        tree = cKDTree(target_pts)
        R_total = np.eye(2)
        t_total = np.zeros(2)
        prev_error = float('inf')

        for _ in range(max_iter):
            distances, indices = tree.query(src)
            matched = target_pts[indices]
            src_c = src.mean(axis=0)
            tgt_c = matched.mean(axis=0)
            H_mat = (src - src_c).T @ (matched - tgt_c)
            U, _, Vt = np.linalg.svd(H_mat)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            candidate_R = R @ R_total
            if abs(np.degrees(np.arctan2(candidate_R[1, 0], candidate_R[0, 0]))) > max_rotation_deg:
                R = np.eye(2)
            t = tgt_c - R @ src_c
            src = (R @ src.T).T + t
            R_total = R @ R_total
            t_total = R @ t_total + t
            error = distances.mean()
            if abs(prev_error - error) < tol:
                break
            prev_error = error

        # Report error on original (non-augmented) contour points only
        final_src = (R_total @ source_pts.T).T + t_total
        final_dist, _ = tree.query(final_src)
        return R_total, t_total, float(final_dist.mean())

    def calculate(self):
        affected_mask          = self.data['affected_mask']
        transformed_unaff_mask = self.data['transformed_unaff_mask']
        affected_laterality    = self.data['affected_laterality']
        unaff_width            = self.analyzer.eq_measurements['unaff_width']

        aff_neck_mask   = self.data.get('affected_neck_mask')
        unaff_neck_mask = self.data.get('transformed_unaff_neck_mask')
        neck_available  = (aff_neck_mask is not None and aff_neck_mask.any()
                           and unaff_neck_mask is not None and unaff_neck_mask.any())

        icp_error   = float('nan')
        icp_success = False
        alignment_mode = 'head_only_icp'

        try:
            if neck_available:
                # ── Neck-enhanced alignment ──────────────────────────────────────
                # Step 1: 3-point landmark pre-alignment
                aff_lm  = self.extract_medial_column_landmarks(
                    affected_mask, aff_neck_mask, affected_laterality)
                unaff_lm = self.extract_medial_column_landmarks(
                    transformed_unaff_mask, unaff_neck_mask, affected_laterality)

                if aff_lm is not None and unaff_lm is not None:
                    R_pre, t_pre = self.compute_rigid_from_landmarks(unaff_lm, aff_lm)
                    # Cap pre-alignment rotation to 20 degrees
                    pre_angle = abs(np.degrees(np.arctan2(R_pre[1, 0], R_pre[0, 0])))
                    if pre_angle > 20:
                        R_pre, t_pre = np.eye(2), np.zeros(2)
                    pre_aligned_head = self._apply_rigid_transform_to_mask(
                        transformed_unaff_mask, R_pre, t_pre)
                    pre_aligned_neck = self._apply_rigid_transform_to_mask(
                        unaff_neck_mask, R_pre, t_pre)
                    landmark_src = (R_pre @ unaff_lm.T).T + t_pre
                else:
                    R_pre, t_pre = np.eye(2), np.zeros(2)
                    pre_aligned_head = transformed_unaff_mask
                    pre_aligned_neck = unaff_neck_mask
                    landmark_src = None

                # Step 2: Weighted ICP on combined head+neck contours
                combined_src_mask = np.logical_or(pre_aligned_head, pre_aligned_neck)
                combined_tgt_mask = np.logical_or(affected_mask, aff_neck_mask)
                src_pts = self._extract_contour_points(combined_src_mask)
                tgt_pts = self._extract_contour_points(combined_tgt_mask)

                R_icp, t_icp, icp_error = self.icp_align_weighted(
                    src_pts, tgt_pts, landmark_pts=landmark_src)

                # Compose pre-alignment + ICP
                R = R_icp @ R_pre
                t = R_icp @ t_pre + t_icp
                icp_aligned = self._apply_rigid_transform_to_mask(
                    transformed_unaff_mask, R, t)

                if icp_aligned.sum() >= 0.5 * transformed_unaff_mask.sum():
                    icp_success    = True
                    alignment_mode = 'neck_landmark_icp'
                else:
                    print("Neck ICP: aligned mask too sparse, falling back to head-only ICP")

            if not icp_success:
                # ── Head-only ICP (original behaviour) ───────────────────────────
                R, t, icp_error = self.icp_align(transformed_unaff_mask, affected_mask)
                icp_aligned = self._apply_rigid_transform_to_mask(
                    transformed_unaff_mask, R, t)
                if icp_aligned.sum() >= 0.5 * transformed_unaff_mask.sum():
                    icp_success    = True
                    alignment_mode = 'head_only_icp'
                else:
                    print("ICP: aligned mask too sparse, falling back to landmark alignment")

        except Exception as e:
            print(f"ICP alignment failed ({e}), using landmark-based fallback")

        if icp_success:
            aff_padded  = affected_mask.copy()
            unaff_padded = icp_aligned
        else:
            # Fallback: original landmark-based translation with padded canvas
            aff_landmark  = self.find_landmark(affected_mask, affected_laterality)
            unaff_landmark = self.find_landmark(transformed_unaff_mask, affected_laterality)
            dx = int(round(aff_landmark[0] - unaff_landmark[0]))
            dy = int(round(aff_landmark[1] - unaff_landmark[1]))
            H_orig, W_orig = affected_mask.shape
            min_x = min(0, dx)
            max_x = max(W_orig - 1, dx + W_orig - 1)
            min_y = min(0, dy)
            max_y = max(H_orig - 1, dy + H_orig - 1)
            new_width  = max_x - min_x + 1
            new_height = max_y - min_y + 1
            aff_padded  = np.zeros((new_height, new_width), dtype=bool)
            unaff_padded = np.zeros((new_height, new_width), dtype=bool)
            aff_padded[ -min_y:-min_y + H_orig, -min_x:-min_x + W_orig] = affected_mask
            unaff_padded[dy - min_y:dy - min_y + H_orig, dx - min_x:dx - min_x + W_orig] = transformed_unaff_mask
            alignment_mode = 'landmark_fallback'

        # Compute boundary profiles
        top_aff,   bottom_aff,  left_aff,   right_aff   = self.compute_boundaries(aff_padded)
        top_unaff, bottom_unaff, left_unaff, right_unaff = self.compute_boundaries(unaff_padded)

        canvas_h, canvas_w = aff_padded.shape

        # Calculate max height difference (ΔH)
        max_diff_top = max_diff_bottom = 0
        for x in range(canvas_w):
            if top_aff[x] != -1 and top_unaff[x] != -1:
                max_diff_top    = max(max_diff_top,    abs(top_aff[x]    - top_unaff[x]))
            if bottom_aff[x] != -1 and bottom_unaff[x] != -1:
                max_diff_bottom = max(max_diff_bottom, abs(bottom_aff[x] - bottom_unaff[x]))
        deltaH = max(max_diff_top, max_diff_bottom)

        # Calculate max width difference (ΔW)
        max_diff_left = max_diff_right = 0
        for y in range(canvas_h):
            if left_aff[y] != -1 and left_unaff[y] != -1:
                max_diff_left  = max(max_diff_left,  abs(left_aff[y]  - left_unaff[y]))
            if right_aff[y] != -1 and right_unaff[y] != -1:
                max_diff_right = max(max_diff_right, abs(right_aff[y] - right_unaff[y]))
        deltaW = max(max_diff_left, max_diff_right)

        deformity_index = (deltaH + deltaW) / unaff_width
        self.analyzer.di_measurements = {
            'deformity_index':        deformity_index,
            'deltaH':                 deltaH,
            'deltaW':                 deltaW,
            'unaff_diameter':         unaff_width,
            'aff_padded':             aff_padded,
            'unaff_padded':           unaff_padded,
            'icp_registration_error': icp_error,
            'alignment_mode':         alignment_mode,
        }