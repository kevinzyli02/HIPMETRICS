# --- CONFIG ---
INPUT = "HerringRepository_KevinLiReport_PATIENT_LEVEL.csv"
BAR_PNG = "lateral_pillar_fragmentation_bar.png"
FILTERED_CSV = "patients_with_lateral_pillar_fragmentation.csv"

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment this if you're in a notebook
import matplotlib.pyplot as plt
import re

def normalize_colname(c: str) -> str:
    if not isinstance(c, str):
        return c
    c2 = re.sub(r"[\u0000-\u001F\u007F\u00A0]+", " ", c)
    return c2.strip()

def canon(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())

# target column fuzzy match
needle_base = "Lateral Pillar Classification in Fragmentation Stage"
needle_canon = canon(needle_base)

df = pd.read_csv(INPUT, dtype=str)
df.rename(columns=lambda c: normalize_colname(c), inplace=True)
cols = list(df.columns)

# find the classification column robustly
lp_cols = [c for c in cols if canon(c) == needle_canon]
if not lp_cols:
    lp_cols = [c for c in cols if needle_canon in canon(c)]
if not lp_cols:
    raise RuntimeError("Could not find the Lateral Pillar Classification (Fragmentation) column.")
lp_col = lp_cols[0]

pid_col = "Record ID" if "Record ID" in cols else cols[0]

# normalize class values and pick the most severe per patient
severity_rank = {"A": 0, "B": 1, "B/C": 2, "C": 3}

def normalize_class(val: str):
    if val is None:
        return None
    s = str(val).strip().upper()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", "/").replace("\\", "/").replace(" ", "")
    if s in {"", "NA", "N/A", "NONE", "UNKNOWN", "UNCLASSIFIABLE", "NOTAVAILABLE"}:
        return None
    if s in {"A", "B", "C"}:
        return s
    if s.startswith("B/C") or s in {"B/C", "BC", "B C", "B/CBORDERLINE", "B/CBORDERLINE?", "B/C?"}:
        return "B/C"
    if s in {"AB", "A/B"}:
        return "B"
    if s in {"AC", "A/C"}:
        return "C"
    return None

# Each row may have multiple values joined by ';'
chosen = []
classes_all = []
for raw in df[lp_col].fillna(""):
    parts = [p.strip() for p in str(raw).split(";") if p is not None]
    norm = [normalize_class(p) for p in parts]
    norm = [x for x in norm if x is not None]
    classes_all.append("; ".join(sorted(set(norm), key=lambda x: severity_rank.get(x, -1))))
    if norm:
        c = sorted(set(norm), key=lambda x: severity_rank.get(x, -1))[-1]  # most severe
    else:
        c = None
    chosen.append(c)

out = df[[pid_col]].copy()
out["Chosen Lateral Pillar Class (Fragmentation)"] = chosen
out["All Classes Found"] = classes_all
out = out[out["Chosen Lateral Pillar Class (Fragmentation)"].notna()]

# counts + bar chart
order = ["A", "B", "B/C", "C"]
counts = out["Chosen Lateral Pillar Class (Fragmentation)"].value_counts().reindex(order).fillna(0).astype(int)

plt.figure(figsize=(8, 5))
ax = counts.plot(kind="bar", color=["#6baed6", "#3182bd", "#9e9ac8", "#756bb1"])
ax.set_title("Lateral Pillar Classification (Fragmentation Stage) — Patient-Level (Most Severe)")
ax.set_xlabel("Classification")
ax.set_ylabel("Number of Patients")
ax.set_xticklabels(order, rotation=0)
for i, v in enumerate(counts.values):
    ax.text(i, v + max(1, counts.max() * 0.01), str(v), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(BAR_PNG, dpi=160)
plt.close()

out.to_csv(FILTERED_CSV, index=False)

print("Patients with classification:", len(out))
print("Counts by class:\n", counts)
print("Saved chart:", BAR_PNG)
