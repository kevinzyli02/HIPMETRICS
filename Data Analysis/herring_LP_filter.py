# --- CONFIG ---
INPUT = r"Y:\Clinical Research\KIM\Kim_1443 Herring Repository\PROJECTS & SUBSTUDIES\Kevin Li Project\HerringRepository_filtered.csv"
BAR_PNG = r"Y:\Clinical Research\KIM\Kim_1443 Herring Repository\PROJECTS & SUBSTUDIES\Kevin Li Project\lateral_pillar_fragmentation_bar.png"
FILTERED_CSV = r"Y:\Clinical Research\KIM\Kim_1443 Herring Repository\PROJECTS & SUBSTUDIES\Kevin Li Project\patients_with_lateral_pillar_fragmentation.csv"

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment this out if you're running in a notebook
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# ----------------- helpers -----------------
def normalize_colname(c: str) -> str:
    """Strip control chars / NBSP and trim so fuzzy name matches work."""
    if not isinstance(c, str):
        return c
    c2 = re.sub(r"[\u0000-\u001F\u007F\u00A0]+", " ", c)
    return c2.strip()

def canon(s: str) -> str:
    """Lowercase + remove whitespace for tolerant matching."""
    return re.sub(r"\s+", "", s.strip().lower())

# normalization for classes -> {A, B, B/C, C}
severity_rank = {"A": 0, "B": 1, "B/C": 2, "C": 3}
def normalize_class(val: str):
    if val is None:
        return None
    s = str(val).strip().upper()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    t = s.replace("-", "/").replace("\\", "/").replace(" ", "")
    if t in {"", "NA", "N/A", "NONE", "UNKNOWN", "UNCLASSIFIABLE", "NOTAVAILABLE"}:
        return None
    if t in {"A", "B", "C"}:
        return t
    if "B/C" in t or t == "BC" or "BORDERLINE" in s:
        return "B/C"
    if t in {"AB", "A/B"}:
        return "B"
    if t in {"AC", "A/C"}:
        return "C"
    # heuristics
    if ("B" in t and "C" in t):
        return "B/C"
    if "C" in t:
        return "C"
    if "B" in t:
        return "B"
    if "A" in t:
        return "A"
    return None

# 🔧 CHANGES: helper to resolve fuzzy column names once
def resolve_col(cols, target_name, fallback_contains=False):
    """
    Return the best-matching column name from cols for the target_name.
    First tries exact canonical match; if not found and fallback_contains=True,
    tries 'contains' on canonical names.
    """
    target = canon(target_name)
    exact = [c for c in cols if canon(c) == target]
    if exact:
        return exact[0]
    if fallback_contains:
        contains = [c for c in cols if target in canon(c)]
        if contains:
            return contains[0]
    return None

# ----------------- load filtered CSV -----------------
df = pd.read_csv(INPUT, dtype=str, low_memory=False)
df.rename(columns=lambda c: normalize_colname(c), inplace=True)
cols = list(df.columns)

# patient key: prefer Record ID (since LCPD Study ID is empty)
pid_col = "Record ID" if "Record ID" in cols else cols[0]

# keep LCPD Study ID column if present (even if empty) to output alongside
lcpd_col = "LCPD Study ID" if "LCPD Study ID" in cols else None

# find a classification column:
#   1) prefer consolidated 'Chosen Lateral Pillar Class (Fragmentation)'
#   2) else fall back to 'Lateral Pillar Classification in Fragmentation Stage'
cand1 = [c for c in cols if canon(c) == canon("Chosen Lateral Pillar Class (Fragmentation)")]
if cand1:
    class_col = cand1[0]
else:
    needle = canon("Lateral Pillar Classification in Fragmentation Stage")
    cand2 = [c for c in cols if canon(c) == needle] or [c for c in cols if needle in canon(c)]
    if not cand2:
        raise RuntimeError("Could not find a lateral pillar classification column in the filtered CSV.")
    class_col = cand2[0]

print(f"[INFO] Using patient key: {pid_col!r}")
if lcpd_col:
    print(f"[INFO] Found LCPD column: {lcpd_col!r} (values may be empty)")
print(f"[INFO] Using classification column: {class_col!r}")

# 🔧 CHANGES: resolve extra fields you want carried into the output
extra_wanted = [
    "Sex",
    "Laterality (Hip Code)",
    "Date of Birth",
]
# try exact, then contains
resolved_extra = {}
for name in extra_wanted:
    colname = resolve_col(cols, name, fallback_contains=True)
    if colname:
        resolved_extra[name] = colname
    else:
        print(f"[WARN] Could not find column for: {name!r}; will output as empty.")

# ----------------- aggregate per patient (most severe) -----------------
pid_to_classes = defaultdict(set)
pid_to_lcpd = {}
# 🔧 CHANGES: store first non-empty value per patient for the extra fields
pid_to_extra = {name: {} for name in extra_wanted}  # dict of {wanted_name: {pid: value}}

for _, row in df.iterrows():
    pid = row.get(pid_col)
    if not isinstance(pid, str) or not pid.strip():
        continue

    # capture LCPD Study ID if present
    if lcpd_col:
        lcpd_val = row.get(lcpd_col)
        if isinstance(lcpd_val, str) and lcpd_val.strip() and pid not in pid_to_lcpd:
            pid_to_lcpd[pid] = lcpd_val.strip()

    # 🔧 CHANGES: capture extra fields (first non-empty per pid)
    for wanted_name, colname in resolved_extra.items():
        if colname is None:
            continue
        v = row.get(colname)
        if isinstance(v, str):
            v2 = v.strip()
            if v2 and pid not in pid_to_extra[wanted_name]:
                pid_to_extra[wanted_name][pid] = v2

    raw = row.get(class_col)
    if raw is None or str(raw).strip() == "":
        continue

    # handle semicolon-joined values (from consolidated sources)
    parts = [p.strip() for p in str(raw).split(";")] if ";" in str(raw) else [str(raw).strip()]
    for p in parts:
        nc = normalize_class(p)
        if nc is not None:
            pid_to_classes[pid].add(nc)

# build rows
rows = []
for pid, classes in pid_to_classes.items():
    if not classes:
        continue
    chosen = sorted(classes, key=lambda x: severity_rank.get(x, -1))[-1]
    rec = {
        pid_col: pid,
        "Chosen Lateral Pillar Class (Fragmentation)": chosen,
        "All Classes Found": "; ".join(sorted(classes, key=lambda x: severity_rank.get(x, -1)))
    }
    if lcpd_col:
        rec["LCPD Study ID"] = pid_to_lcpd.get(pid, "")

    # 🔧 CHANGES: add extra fields (empty string if not found for this pid)
    for wanted_name in extra_wanted:
        rec[wanted_name] = pid_to_extra[wanted_name].get(pid, "")

    rows.append(rec)

out = pd.DataFrame(rows)

# reorder columns for readability
# 🔧 CHANGES: bring the new fields to the front after ID(s)
front = [c for c in [pid_col, "LCPD Study ID",
                     "Sex", "Laterality (Hip Code)", "Date of Birth",
                     "Chosen Lateral Pillar Class (Fragmentation)",
                     "All Classes Found"] if c in out.columns]
others = [c for c in out.columns if c not in front]
out = out[front + others] if out.shape[0] > 0 else pd.DataFrame(columns=front + others)

# ----------------- chart -----------------
order = ["A", "B", "B/C", "C"]
if not out.empty and "Chosen Lateral Pillar Class (Fragmentation)" in out.columns:
    counts = out["Chosen Lateral Pillar Class (Fragmentation)"].value_counts().reindex(order).fillna(0).astype(int)

    plt.figure(figsize=(8, 5))
    ax = counts.plot(kind="bar", color=["#6baed6", "#3182bd", "#9e9ac8", "#756bb1"])
    ax.set_title("Lateral Pillar Classification (Fragmentation Stage) — Patient-Level (Most Severe)")
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Patients")
    ax.set_xticklabels(order, rotation=0)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(1, counts.max()*0.01), str(v), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(BAR_PNG, dpi=160)
    plt.close()

    print("Patients with classification:", len(out))
    print("Counts by class:\n", counts)
    print("Saved chart:", BAR_PNG)
else:
    print("[WARN] No patients with recognizable Lateral Pillar classes; chart not created.")



# ----------------- stats: sex, laterality, L/R breakdown -----------------
def normalize_sex(val: str) -> str:
    if not isinstance(val, str):
        return "Unknown"
    s = val.strip().lower()
    if s in {"m", "male", "man"}:
        return "Male"
    if s in {"f", "female", "woman"}:
        return "Female"
    if s in {"other", "nonbinary", "non-binary", "nb"}:
        return "Other"
    return "Unknown"

def parse_laterality(val: str):
    """
    Parse laterality strings that may be formatted as 'Laterality,Side',
    such as 'Bilateral, Both' or 'Unilateral, Left/Right'.

    Returns a tuple: (lat_class, side)
      lat_class in {"Unilateral", "Bilateral", "Unknown"}
      side in {"L", "R", None}
    For Bilateral, side is None (counted to both sides in per-hip stats).
    """

    def norm_token(x: str) -> str:
        return re.sub(r"\s+", " ", x.strip().lower()) if isinstance(x, str) else ""

    def map_side(s: str):
        s = norm_token(s)
        if s in {"l", "left"}:
            return "L"
        if s in {"r", "right"}:
            return "R"
        if s in {"both", "bilateral", "b/l", "b\\l"}:
            return None  # represent bilateral as None for side
        return None

    if not isinstance(val, str) or not val.strip():
        return ("Unknown", None)

    s_raw = val.replace("\u00A0", " ").strip()  # normalize NBSPs
    # First, check for a comma-separated "Laterality,Side" format
    if "," in s_raw:
        parts = [p.strip() for p in s_raw.split(",")]
        # Only care about the first two fields; discard extras if present
        lat_tok = norm_token(parts[0]) if len(parts) >= 1 else ""
        side_tok = norm_token(parts[1]) if len(parts) >= 2 else ""

        # Decide laterality class
        if lat_tok in {"bilateral", "bilat", "b"} or "bilat" in lat_tok or "both" in lat_tok:
            return ("Bilateral", None)

        if lat_tok in {"unilateral", "uni", "u"}:
            side = map_side(side_tok)
            # If side couldn't be mapped, keep Unilateral but unknown side
            if side in {"L", "R"}:
                return ("Unilateral", side)
            else:
                return ("Unilateral", None)

        # If the first token didn't clearly indicate laterality, try interpreting the side
        side = map_side(side_tok)
        if side in {"L", "R"}:
            return ("Unilateral", side)
        if side is None and side_tok:  # had a side token but mapped to None as 'both'
            return ("Bilateral", None)

        # Fall through to free-form parsing below

    # ---------- Free-form / legacy parsing (single field) ----------
    s = norm_token(s_raw)

    # Explicit bilateral phrases
    if "bilat" in s or "both" in s or "b/l" in s or "b\\l" in s:
        return ("Bilateral", None)
    if s in {"bilateral", "b"}:
        return ("Bilateral", None)

    # Remove punctuation for LR combos like "L+R", "R/L"
    t = re.sub(r"[^a-z0-9]+", "", s)

    # Common unilateral codes/words
    if t in {"l", "left"}:
        return ("Unilateral", "L")
    if t in {"r", "right"}:
        return ("Unilateral", "R")

    # Combos like "lr", "rl"
    if ("l" in t) and ("r" in t):
        return ("Bilateral", None)

    # Single-letter residues
    if "l" in t and "r" not in t:
        return ("Unilateral", "L")
    if "r" in t and "l" not in t:
        return ("Unilateral", "R")

    return ("Unknown", None)


# Make safe copies of the columns (they already exist in `out` from earlier step)
sex_col = "Sex" if "Sex" in out.columns else None
lat_col = "Laterality (Hip Code)" if "Laterality (Hip Code)" in out.columns else None

# Normalize Sex into a new column for counting
if sex_col:
    out["_SexNorm"] = out[sex_col].apply(normalize_sex)
else:
    out["_SexNorm"] = "Unknown"

# Parse laterality into structured columns
if lat_col:
    parsed = out[lat_col].apply(parse_laterality)
else:
    parsed = [("Unknown", None)] * len(out)

out["_LatClass"] = [lc for lc, _ in parsed]  # "Unilateral" / "Bilateral" / "Unknown"
out["_Side"] = [sd for _, sd in parsed]      # "L" / "R" / None

# ---- Sex counts
sex_counts = out["_SexNorm"].value_counts(dropna=False).sort_index()

# ---- Laterality counts
lat_counts = out["_LatClass"].value_counts(dropna=False).sort_index()

# ---- L/R breakdowns
# (a) Per-patient (unilateral only)
unilateral = out[out["_LatClass"] == "Unilateral"]
lr_per_patient = unilateral["_Side"].value_counts(dropna=False).reindex(["L", "R"], fill_value=0)

# (b) Per-hip (count bilaterals to both L and R)
per_hip_counts = {"L": 0, "R": 0}
for lc, side in zip(out["_LatClass"], out["_Side"]):
    if lc == "Unilateral":
        if side in {"L", "R"}:
            per_hip_counts[side] += 1
    elif lc == "Bilateral":
        per_hip_counts["L"] += 1
        per_hip_counts["R"] += 1
# Convert to a printable series-like object
lr_per_hip_L = per_hip_counts["L"]
lr_per_hip_R = per_hip_counts["R"]

# ---- Print the stats
print("\n=== Summary Stats ===")
print("Sex counts:")
for k in ["Female", "Male", "Other", "Unknown"]:
    if k in sex_counts.index:
        print(f"  {k}: {int(sex_counts[k])}")

print("\nLaterality (patient-level):")
for k in ["Unilateral", "Bilateral", "Unknown"]:
    if k in lat_counts.index:
        print(f"  {k}: {int(lat_counts[k])}")

print("\nLeft/Right breakdown (per-patient, unilateral only):")
print(f"  Left: {int(lr_per_patient.get('L', 0))}")
print(f"  Right: {int(lr_per_patient.get('R', 0))}")

print("\nLeft/Right breakdown (per-hip, counting bilaterals to both sides):")
print(f"  Left: {lr_per_hip_L}")
print(f"  Right: {lr_per_hip_R}")
print("=== End Summary Stats ===\n")

# ----------------- ensure _LatClass/_Side exist, then filter to unilateral -----------------
# If you haven't already created _LatClass/_Side via the stats section, compute them now:
if "_LatClass" not in out.columns or "_Side" not in out.columns:
    def normalize_sex(val: str) -> str:
        if not isinstance(val, str):
            return "Unknown"
        s = val.strip().lower()
        if s in {"m", "male", "man"}: return "Male"
        if s in {"f", "female", "woman"}: return "Female"
        if s in {"other", "nonbinary", "non-binary", "nb"}: return "Other"
        return "Unknown"


    def parse_laterality(val: str):
        # Use your latest version that supports "Laterality,Side" (Bilateral, Both / Unilateral, Left/Right)
        def norm_token(x: str) -> str:
            return re.sub(r"\s+", " ", x.strip().lower()) if isinstance(x, str) else ""

        def map_side(s: str):
            s = norm_token(s)
            if s in {"l", "left"}: return "L"
            if s in {"r", "right"}: return "R"
            if s in {"both", "bilateral", "b/l", "b\\l"}: return None
            return None

        if not isinstance(val, str) or not val.strip():
            return ("Unknown", None)
        s_raw = val.replace("\u00A0", " ").strip()
        if "," in s_raw:
            parts = [p.strip() for p in s_raw.split(",")]
            lat_tok = norm_token(parts[0]) if len(parts) >= 1 else ""
            side_tok = norm_token(parts[1]) if len(parts) >= 2 else ""
            if lat_tok in {"bilateral", "bilat", "b"} or "bilat" in lat_tok or "both" in lat_tok:
                return ("Bilateral", None)
            if lat_tok in {"unilateral", "uni", "u"}:
                side = map_side(side_tok)
                return ("Unilateral", side if side in {"L", "R"} else None)
            side = map_side(side_tok)
            if side in {"L", "R"}: return ("Unilateral", side)

# ----------------- save table -----------------
out.to_csv(FILTERED_CSV, index=False)
print("Saved filtered list:", FILTERED_CSV)