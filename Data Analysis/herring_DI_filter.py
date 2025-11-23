# --- CONFIG ---

RAW_INPUT = r"Y:\Clinical Research\KIM\Kim_1443 Herring Repository\PROJECTS & SUBSTUDIES\Kevin Li Project\HerringRepository-KevinLiReport_DATA_LABELS_2025-09-29_1426.csv"
ALT_INPUT = r"Y:\Clinical Research\KIM\Kim_1443 Herring Repository\PROJECTS & SUBSTUDIES\Kevin Li Project\HerringRepository_filtered.csv"
OUTPUT_CSV = r"Y:\Clinical Research\KIM\Kim_1443 Herring Repository\PROJECTS & SUBSTUDIES\Kevin Li Project\two_year_followup.csv"

import os, re, gc
import pandas as pd
from collections import defaultdict
from pandas.tseries.offsets import DateOffset

# ----------------- helpers -----------------
def normalize_colname(c: str) -> str:
    """Strip control chars / NBSP and trim so fuzzy name matches work."""
    if not isinstance(c, str): return c
    c2 = re.sub(r"[\u0000-\u001F\u007F\u00A0]+", " ", c)
    return c2.strip()

def canon(s: str) -> str:
    """Lowercase + remove whitespace for tolerant matching."""
    return re.sub(r"\s+", "", s.strip().lower())

def in_two_year_window(induction: pd.Timestamp, xray: pd.Timestamp) -> bool:
    """Exact ±6 calendar months around induction + 2 years."""
    if pd.isna(induction) or pd.isna(xray): return False
    target = induction + DateOffset(years=2)
    lower = target - DateOffset(months=6)
    upper = target + DateOffset(months=6)
    return lower <= xray <= upper

def abs_days_to_two_year_target(induction: pd.Timestamp, xray: pd.Timestamp) -> float:
    target = induction + DateOffset(years=2)
    return abs((xray - target).days)

# ----------------- choose input -----------------
input_path = RAW_INPUT if os.path.exists(RAW_INPUT) else (ALT_INPUT if os.path.exists(ALT_INPUT) else None)
if input_path is None:
    raise SystemExit(
        "Input file not found. Please place one of these in the working folder:\n"
        f" - {RAW_INPUT}\n"
        f" - {ALT_INPUT}\n"
    )

print("Reading:", input_path)

# ----------------- inspect columns -----------------
sample = pd.read_csv(input_path, nrows=200, dtype=str, low_memory=False)
sample.rename(columns=lambda c: normalize_colname(c), inplace=True)
cols = list(sample.columns)

# patient key
pid_col = "Record ID" if "Record ID" in cols else ("LCPD Study ID" if "LCPD Study ID" in cols else cols[0])
lcpd_col = "LCPD Study ID" if "LCPD Study ID" in cols else cols[0]
# Date of Induction (handle variants/whitespace)
ind_col_candidates = [c for c in cols if canon(c) == canon("Date of Induction")] \
                  or [c for c in cols if "dateofinduction" in canon(c)]
if not ind_col_candidates:
    raise SystemExit("Could not find 'Date of Induction' column.")
ind_col = ind_col_candidates[0]

# X-ray date columns: Date of X-Ray1..10 (support any count)
xr_cols = [c for c in cols if re.match(r"(?i)^\s*Date\s*of\s*X-?Ray\s*\d+\s*$", c)]
xr_cols = sorted(xr_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))
if not xr_cols:
    raise SystemExit("Could not find any 'Date of X-Ray#' columns.")

# ----------------- aggregate per patient -----------------
ind_by_pid: dict = {}                # earliest non-null induction per patient
xr_by_pid: defaultdict = defaultdict(set)   # all x-ray dates per patient

for chunk in pd.read_csv(input_path, chunksize=50_000, dtype=str, low_memory=False):
    chunk.rename(columns=lambda c: normalize_colname(c), inplace=True)
    keep = [pid_col, ind_col] + [c for c in xr_cols if c in chunk.columns]
    sub = chunk[keep]
    sub = sub[sub[pid_col].notna()]

    for _, row in sub.iterrows():
        pid = row[pid_col]

        # induction date: keep earliest if multiple rows
        ind_val = pd.to_datetime(row.get(ind_col), errors="coerce", infer_datetime_format=True)
        if not pd.isna(ind_val):
            if pid not in ind_by_pid or ind_val < ind_by_pid[pid]:
                ind_by_pid[pid] = ind_val

        # x-ray dates: collect from all XR columns
        for xc in xr_cols:
            xv = pd.to_datetime(row.get(xc), errors="coerce", infer_datetime_format=True)
            if not pd.isna(xv):
                xr_by_pid[pid].add(xv)

    del chunk, sub
    gc.collect()

# ----------------- choose 2Y follow-up per patient -----------------
records = []
for pid, xr_set in xr_by_pid.items():
    induction = ind_by_pid.get(pid, pd.NaT)
    if pd.isna(induction) or not xr_set:
        continue

    # keep only x-rays within ±6m of 2y
    in_window = [x for x in xr_set if in_two_year_window(induction, x)]
    if not in_window:
        continue

    # pick the closest to exactly 2 years
    best = min(in_window, key=lambda x: abs_days_to_two_year_target(induction, x))
    diff_days = abs_days_to_two_year_target(induction, best)
    diff_months = round(diff_days / 30.4375, 2)  # ~months

    records.append({
        pid_col: pid,
        "Perthes Study ID": lcpd_col,
        "Date of Induction": induction.date().isoformat(),
        "2Y Follow-up X-Ray Date": best.date().isoformat(),
        "Abs Diff to 2Y (days)": int(diff_days),
        "Abs Diff to 2Y (months, ~)": diff_months
    })

result = pd.DataFrame.from_records(records)

if result.empty:
    print("No patients met the 2-year ±6 months criterion.")
else:
    # Sort by patient and then by closeness to 2y
    result["__pidnum"] = pd.to_numeric(result[pid_col], errors="coerce")
    result.sort_values(["__pidnum", "Abs Diff to 2Y (days)"], inplace=True)
    result.drop(columns=["__pidnum"], inplace=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print("Patients meeting 2-year ±6 months window:", len(result))
    print("Saved:", OUTPUT_CSV)
    print(result.head(10).to_string(index=False))
