"""
ICC Analysis: HIPMETRICS algorithm vs ground truth measurements.

Computes:
  - ICC(2,1) inter-rater (manual vs manual) when two raters are available
  - ICC(3,1) method-comparison (manual vs algorithm)
  - Supplementary: Pearson r, MAE, RMSE, Bland-Altman bias/LoA

To add a metric (e.g. lateral pillar):
  1. Add its columns to the ground truth Excel file
  2. Add an entry to METRICS below
  3. Re-run — everything else is automatic

To enable inter-rater ICC for a metric, set its 'rater2' key to the
corresponding column name in the parsed ground truth DataFrame.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# ─── Paths ────────────────────────────────────────────────────────────────────

GT_FILE = Path(
    '/Users/kevin/Library/CloudStorage/OneDrive-ScottishRiteforChildren'
    '/Kim Research/HIPMETRICS-lateral pillar/DI images/DI Measurements.xlsx'
)
COMPUTED_FILE = Path(
    '/Users/kevin/Documents/GitHub/HIPMETRICS/Data Analysis/measurements_summary.xlsx'
)
SHEET = 'Test'   # 'Test' or 'Train'
OUTPUT_FILE = Path(
    '/Users/kevin/Documents/GitHub/HIPMETRICS/Data Analysis/icc_results.xlsx'
)

# ─── Metric configuration ─────────────────────────────────────────────────────
#
# 'rater1'   : column name in the parsed ground truth DataFrame (see load_ground_truth)
# 'rater2'   : column name for second rater, or None if not yet collected
# 'computed' : column name in measurements_summary.xlsx produced by main.py
#
METRICS = {
    'EI (Epiphyseal Index)': {
        'rater1':   'EI_r1',
        'rater2':   None,           # set to e.g. 'EI_r2' once second-rater data is added
        'computed': 'eq_ratio',
    },
    'DI (Deformity Index)': {
        'rater1':   'DI_r1',
        'rater2':   None,           # set to e.g. 'DI_r2' once second-rater data is added
        'computed': 'deformity_index',
    },
    # ── Uncomment when lateral pillar data is added to the Excel ──────────────
    # 'Lateral Pillar Ratio': {
    #     'rater1':   'LP_r1',
    #     'rater2':   None,
    #     'computed': 'lateral_ratio',
    # },
}


# ─── Ground truth loader ──────────────────────────────────────────────────────

def load_ground_truth(path: Path, sheet: str) -> pd.DataFrame:
    """
    Parse the measurement Excel file.

    Expected layout (two header rows before data):
      Row 1  – group labels: IPSG Number | EQ [merged] | ... | DI [merged] | ...
      Row 2  – column names: IPSG Number | D(aff) | L(aff) | L(unaff) | EI | d | w | h | DI
      Row 3+ – numeric data

    When a second rater's EI/DI columns are appended to the right, extend the
    `names` list below and add matching entries to METRICS.
    """
    names = [
        'patient_id',
        'aff_diameter',     # EQ group: D (affected)
        'aff_height',       # EQ group: L (affected)
        'unaff_height',     # EQ group: L (unaffected)
        'EI_r1',            # EQ group: EI (rater 1)
        'unaff_diameter',   # DI group: d
        'deltaW',           # DI group: w
        'deltaH',           # DI group: h
        'DI_r1',            # DI group: DI (rater 1)
        # Future columns — add here as new raters / metrics are entered:
        # 'EI_r2',
        # 'DI_r2',
        # 'LP_r1',
        # 'LP_r2',
    ]
    df = pd.read_excel(path, sheet_name=sheet, skiprows=2, header=None,
                       names=names, usecols=range(len(names)))
    df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')
    df = df.dropna(subset=['patient_id'])
    df['patient_id'] = df['patient_id'].astype(int).astype(str)
    for col in names[1:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_computed(path: Path) -> pd.DataFrame:
    """Load the measurements_summary.xlsx produced by main.py."""
    df = pd.read_excel(path)
    df['patient_id'] = df['patient_id'].astype(str)
    return df


# ─── ICC implementations ──────────────────────────────────────────────────────

def _anova_components(data: np.ndarray):
    """Two-way ANOVA decomposition for an (n_subjects × n_raters) array."""
    n, k = data.shape
    grand_mean = data.mean()
    subject_means = data.mean(axis=1)
    rater_means = data.mean(axis=0)

    SS_B = k * np.sum((subject_means - grand_mean) ** 2)
    SS_W = np.sum((data - subject_means[:, None]) ** 2)
    SS_R = n * np.sum((rater_means - grand_mean) ** 2)
    SS_E = SS_W - SS_R

    MS_B = SS_B / (n - 1)
    MS_R = SS_R / (k - 1)
    MS_E = SS_E / ((n - 1) * (k - 1))
    return n, k, MS_B, MS_R, MS_E


def icc_2_1(y1: np.ndarray, y2: np.ndarray, alpha: float = 0.05):
    """
    ICC(2,1) — two-way random effects, absolute agreement.
    Appropriate for inter-rater reliability (manual vs manual).
    Returns (icc, lower_95ci, upper_95ci).
    Reference: Shrout & Fleiss (1979).
    """
    data = np.column_stack([y1, y2])
    n, k, MS_B, MS_R, MS_E = _anova_components(data)

    icc_val = (MS_B - MS_E) / (
        MS_B + (k - 1) * MS_E + k * (MS_R - MS_E) / n
    )

    # 95% CI — Shrout & Fleiss approximation
    F0 = MS_B / MS_E
    df1, df2 = n - 1, (n - 1) * (k - 1)
    F_U = stats.f.ppf(1 - alpha / 2, df1, df2)
    F_L = stats.f.ppf(alpha / 2, df1, df2)

    FL = F0 / F_U
    FU = F0 / F_L
    correction = k * (MS_R - MS_E) / (n * MS_E)
    denom_L = FL + (k - 1) + correction
    denom_U = FU + (k - 1) + correction
    ci_lower = (FL - 1) / denom_L if denom_L > 0 else 0.0
    ci_upper = (FU - 1) / denom_U if denom_U > 0 else 1.0

    return float(icc_val), float(ci_lower), float(ci_upper)


def icc_3_1(y1: np.ndarray, y2: np.ndarray, alpha: float = 0.05):
    """
    ICC(3,1) — two-way mixed effects, consistency.
    Appropriate for method comparison (manual vs algorithm).
    Returns (icc, lower_95ci, upper_95ci).
    Reference: McGraw & Wong (1996).
    """
    data = np.column_stack([y1, y2])
    n, k, MS_B, _, MS_E = _anova_components(data)

    icc_val = (MS_B - MS_E) / (MS_B + (k - 1) * MS_E)

    F0 = MS_B / MS_E
    df1, df2 = n - 1, (n - 1) * (k - 1)
    F_U = stats.f.ppf(1 - alpha / 2, df1, df2)
    F_L = stats.f.ppf(alpha / 2, df1, df2)

    ci_lower = (F0 / F_U - 1) / (F0 / F_U + k - 1)
    ci_upper = (F0 / F_L - 1) / (F0 / F_L + k - 1)

    return float(icc_val), float(ci_lower), float(ci_upper)


# ─── Per-pair analysis ────────────────────────────────────────────────────────

def analyze_pair(label: str, y_ref: np.ndarray, y_test: np.ndarray, icc_fn) -> dict | None:
    """Compute ICC + supplementary stats for one (ref, test) measurement pair."""
    mask = np.isfinite(y_ref) & np.isfinite(y_test)
    y1, y2 = y_ref[mask], y_test[mask]
    n = len(y1)
    if n < 5:
        print(f"  [SKIP] {label}: only {n} complete pairs (need ≥ 5)")
        return None

    icc_val, ci_lo, ci_hi = icc_fn(y1, y2)
    r, p_r = stats.pearsonr(y1, y2)
    mae  = float(np.abs(y1 - y2).mean())
    rmse = float(np.sqrt(((y1 - y2) ** 2).mean()))

    diff  = y1 - y2
    bias  = float(diff.mean())
    sd    = float(diff.std(ddof=1))
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd

    return {
        'Comparison':        label,
        'N':                 n,
        'ICC':               round(icc_val, 3),
        'ICC 95% CI':        f'[{ci_lo:.3f}, {ci_hi:.3f}]',
        'Pearson r':         round(float(r), 3),
        'p-value':           round(float(p_r), 4),
        'MAE':               round(mae, 4),
        'RMSE':              round(rmse, 4),
        'Bias (mean diff)':  round(bias, 4),
        'SD (diff)':         round(sd, 4),
        'LoA lower':         round(loa_lo, 4),
        'LoA upper':         round(loa_hi, 4),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading ground truth: {GT_FILE.name}  [sheet={SHEET}]")
    gt = load_ground_truth(GT_FILE, SHEET)
    print(f"  {len(gt)} patients loaded")

    if not COMPUTED_FILE.exists():
        raise FileNotFoundError(
            f"Computed results not found: {COMPUTED_FILE}\n"
            "Run main.py first to generate measurements_summary.xlsx."
        )
    print(f"Loading computed results: {COMPUTED_FILE.name}")
    comp = load_computed(COMPUTED_FILE)

    # One computed row per patient (take first if multiple timepoints)
    comp_dedup = comp.drop_duplicates(subset='patient_id', keep='first')
    print(f"  {len(comp_dedup)} unique patients in computed results")

    # Merge on patient_id
    merged = gt.merge(comp_dedup, on='patient_id', how='inner', suffixes=('_gt', '_comp'))
    print(f"  {len(merged)} patients matched between ground truth and computed\n")

    rows = []

    for metric_name, cfg in METRICS.items():
        r1_col = cfg['rater1']
        r2_col = cfg.get('rater2')
        comp_col = cfg['computed']

        print(f"── {metric_name} ──")

        # Check columns exist
        if r1_col not in merged.columns:
            print(f"  [SKIP] rater1 column '{r1_col}' not found in ground truth")
            continue
        if comp_col not in merged.columns:
            print(f"  [SKIP] computed column '{comp_col}' not found in script output")
            continue

        r1 = merged[r1_col].to_numpy(dtype=float)
        comp_vals = merged[comp_col].to_numpy(dtype=float)

        # Inter-rater ICC (manual vs manual) — only when rater 2 is available
        if r2_col and r2_col in merged.columns:
            r2 = merged[r2_col].to_numpy(dtype=float)
            row = analyze_pair(
                f'{metric_name}: Rater 1 vs Rater 2 [ICC(2,1)]',
                r1, r2, icc_2_1
            )
            if row:
                rows.append(row)
                print(f"  Inter-rater  ICC(2,1) = {row['ICC']}  {row['ICC 95% CI']}  n={row['N']}")
        elif r2_col:
            print(f"  [INFO] rater2 column '{r2_col}' configured but not yet in data — skipping inter-rater ICC")
        else:
            print(f"  [INFO] rater2 not configured — skipping inter-rater ICC")

        # Method-comparison ICC (manual vs algorithm)
        row = analyze_pair(
            f'{metric_name}: Manual vs Algorithm [ICC(3,1)]',
            r1, comp_vals, icc_3_1
        )
        if row:
            rows.append(row)
            print(f"  Manual vs Algorithm  ICC(3,1) = {row['ICC']}  {row['ICC 95% CI']}  n={row['N']}")
        print()

    if not rows:
        print("No results to report.")
        return

    results_df = pd.DataFrame(rows)

    # Console table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print()

    # Excel output
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='ICC Results', index=False)

        # Raw paired data per metric for auditing
        for metric_name, cfg in METRICS.items():
            r1_col, comp_col = cfg['rater1'], cfg['computed']
            if r1_col not in merged.columns or comp_col not in merged.columns:
                continue
            keep = ['patient_id', r1_col, comp_col]
            if cfg.get('rater2') and cfg['rater2'] in merged.columns:
                keep.append(cfg['rater2'])
            sheet_name = metric_name[:31]  # Excel 31-char limit
            merged[keep].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
