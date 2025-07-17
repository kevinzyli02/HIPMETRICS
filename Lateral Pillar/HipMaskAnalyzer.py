import numpy as np
import pandas as pd
from skimage.transform import rotate
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class HipMaskAnalyzer:
    """
    Analyze hip masks: rotate to align major axes horizontally,
    measure pillar heights, compute ratios, visualize overlays, and save results.
    """
    def __init__(self, results_list: List[Dict]):
        """
        results_list: list of dicts with keys:
            'patient_id', 'timepoint', 'affected_mask', 'transformed_unaff_mask',
            'aff_major_axis', 'unaff_major_axis', 'affected_laterality'
        """
        self.results_list = results_list
        self.results_df = None

    @staticmethod
    def _rotation_angle(axis: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
        """
        Compute angle (in degrees) to rotate axis to horizontal.
        Axis endpoints: ((x1,y1),(x2,y2)).
        Returns angle such that rotating by -angle makes axis horizontal.
        """
        (x1, y1), (x2, y2) = axis
        dy = y2 - y1
        dx = x2 - x1
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate binary mask by given angle (degrees) around center, preserving shape.
        """
        return rotate(mask.astype(float), -angle, resize=True, order=0, preserve_range=True) > 0.5

    @staticmethod
    def _measure_sections(mask: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Divide mask width into thirds, measure for each section:
        max height and average height.
        Returns dict: {'left': (max, avg), 'middle': ..., 'right': ...}
        """
        H, W = mask.shape
        third = W // 3
        sections = {
            'left': mask[:, 0:third],
            'middle': mask[:, third:2*third],
            'right': mask[:, 2*third: W]
        }
        results = {}
        for name, sec in sections.items():
            heights = []
            # for each column, compute vertical height of mask
            for col in range(sec.shape[1]):
                col_data = sec[:, col]
                ys = np.where(col_data)[0]
                if ys.size:
                    heights.append(float(ys.max() - ys.min() + 1))
            if heights:
                results[name] = (float(np.max(heights)), float(np.mean(heights)))
            else:
                results[name] = (0.0, 0.0)
        return results

    def _map_pillars(self, sections: Dict[str, Tuple[float, float]], laterality: str) -> Dict[str, Tuple[float, float]]:
        """
        Map left/middle/right to medial/middle/lateral pillars depending on laterality of affected head.
        laterality: 'L' or 'R'
        returns {'medial':(...), 'middle':(...), 'lateral':(...)}
        """
        if laterality == 'R':
            # right affected: leftmost is lateral, rightmost is medial
            return {
                'lateral': sections['left'],
                'middle': sections['middle'],
                'medial': sections['right']
            }
        else:
            # left affected: leftmost is medial, rightmost is lateral
            return {
                'medial': sections['left'],
                'middle': sections['middle'],
                'lateral': sections['right']
            }

    def analyze(self) -> pd.DataFrame:
        """
        Perform full analysis on all results and return a pandas DataFrame of measurements and ratios.
        """
        records = []
        for res in self.results_list:
            pid = res['patient_id']
            tp = res['timepoint']
            lat = res['affected_laterality']

            # compute rotation angle
            angle = self._rotation_angle(res['aff_major_axis'])

            # rotate masks
            aff_rot = self._rotate_mask(res['affected_mask'], angle)
            unaff_rot = self._rotate_mask(res['transformed_unaff_mask'], angle)

            # measure sections
            aff_secs = self._measure_sections(aff_rot)
            unaff_secs = self._measure_sections(unaff_rot)

            # map to pillars
            aff_pillars = self._map_pillars(aff_secs, lat)
            unaff_pillars = self._map_pillars(unaff_secs, lat)

            # compute ratios
            ratios = {}
            for pill in ['lateral', 'middle', 'medial']:
                aff_max, aff_avg = aff_pillars[pill]
                unaff_max, unaff_avg = unaff_pillars[pill]
                ratios[f'{pill}_max_ratio'] = aff_max / unaff_max if unaff_max else np.nan
                ratios[f'{pill}_avg_ratio'] = aff_avg / unaff_avg if unaff_avg else np.nan

            # collect record
            rec = {'patient_id': pid, 'timepoint': tp, 'rotation_angle': angle}
            # flatten pillars
            for pill in ['lateral', 'middle', 'medial']:
                rec[f'aff_{pill}_max'] = aff_pillars[pill][0]
                rec[f'aff_{pill}_avg'] = aff_pillars[pill][1]
                rec[f'unaff_{pill}_max'] = unaff_pillars[pill][0]
                rec[f'unaff_{pill}_avg'] = unaff_pillars[pill][1]
                rec[f'{pill}_max_ratio'] = ratios[f'{pill}_max_ratio']
                rec[f'{pill}_avg_ratio'] = ratios[f'{pill}_avg_ratio']

            records.append(rec)

            # visualize
            self._visualize_overlay(aff_rot, unaff_rot, lat, pid, tp)

        df = pd.DataFrame(records)
        self.results_df = df
        return df

    def _visualize_overlay(self, aff_mask: np.ndarray, unaff_mask: np.ndarray,
                           laterality: str, patient_id: str, timepoint: str) -> None:
        """
        Plot overlay of two masks, with dividing lines at thirds and save figure.
        """
        # overlay masks
        H, W = aff_mask.shape
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(aff_mask, cmap='Reds', alpha=0.5)
        ax.imshow(unaff_mask, cmap='Greens', alpha=0.5)
        # draw dividing lines
        third = W // 3
        ax.axvline(third, color='yellow', linestyle='--')
        ax.axvline(2*third, color='yellow', linestyle='--')
        ax.set_title(f'{patient_id} {timepoint} ({laterality})')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{patient_id}_{timepoint}_overlay.png')
        plt.close(fig)

    def save_to_excel(self, filename: str):
        """
        Save the results DataFrame to an Excel file.
        """
        if self.results_df is None:
            raise RuntimeError("No results to save. Run analyze() first.")
        self.results_df.to_excel(filename, index=False)
