#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Institutional 80/20 Split — Patients & Images, FAST
RAW waldenstrom_stage is balanced at the **image level** (all radiographs),
NOT the deduped patient-level representative row.

Also:
- size_penalty = 1.0, image_penalty = 1.0
- IIb/IIIa subset analysis: include ALL IIb/IIIa patients; keep lateral pillar as a row (missing -> 'Not Reported')
- Restores '02_stats_tests' worksheet with KS/chi-square tests (patient-level + image-level Waldenström).
"""

import argparse
import os
import json
import math
import random
import re
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional SciPy for p-values
try:
    from scipy.stats import chi2_contingency, ks_2samp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================
# Defaults (override via CLI)
# =========================
DEFAULT_COLS = {
    "institution": "institution",
    "file_name": "file_name",
    "age": "age_at_xray",
    "gender": "gender",
    "race": "ethnicity",
    "affected": "affected_vs_unaffected",
    "stulberg": "Standardized Stulberg",   # optional; set weight to 0.0 to ignore
    "patient_id": "parsed_ID",
    # RAW waldenstrom used for balancing & analysis
    "waldenstrom_raw": "waldenstrom_stage",
    "lateral_pillar": "lateral_pillar",
}

# =========================
# Weights (updated per request)
# =========================
WEIGHTS = {
    "size_penalty": 1.0,      # patients  test fraction deviation
    "image_penalty": 1.0,     # images    test fraction deviation
    "inst_penalty": 1.0,      # institutions test fraction deviation
    "age": 1.0,
    "gender": 2.0,
    "race": 1.0,
    "affected": 1.0,
    "waldenstrom_raw": 5.0,   # RAW Waldenström balanced on **image-level**
    "stulberg": 0.0,          # set 0.0 to ignore
}

DEFAULT_AGE_BINS = 10


# ======================
# Robust Excel I/O
# ======================
def safe_read_excel(path, sheet_name=None, try_copy=False):
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except PermissionError:
        try:
            with open(path, "rb") as f:
                return pd.read_excel(f, sheet_name=sheet_name, engine="openpyxl")
        except PermissionError:
            if not try_copy:
                raise
            base = os.path.basename(path)
            tmpdir = tempfile.mkdtemp(prefix="onedrive_xls_")
            tmp_path = os.path.join(tmpdir, base)
            shutil.copy2(path, tmp_path)
            try:
                return pd.read_excel(tmp_path, sheet_name=sheet_name, engine="openpyxl")
            finally:
                pass


# ======================
# Helpers & standardizers
# ======================
def coalesce_column(df: pd.DataFrame, name: str):
    if name is None:
        return None
    if name in df.columns:
        return name
    low = str(name).strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == low:
            return c
    raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")

def standardize_gender(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip().lower()
    if s in {"m","male","man"}: return "Male"
    if s in {"f","female","woman"}: return "Female"
    if s in {"other","nonbinary","non-binary","nb"}: return "Other"
    return "Unknown"

def standardize_race_ethnicity(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip().lower()
    mapping = {
        "white":"White","caucasian":"White",
        "black":"Black or African American","african american":"Black or African American",
        "asian":"Asian",
        "native":"American Indian/Alaska Native","american indian":"American Indian/Alaska Native","alaska native":"American Indian/Alaska Native",
        "native hawaiian":"Native Hawaiian/Pacific Islander","pacific islander":"Native Hawaiian/Pacific Islander",
        "two or more":"Two or More","mixed":"Two or More",
        "hispanic":"Hispanic/Latino","latino":"Hispanic/Latino",
        "declined":"Unknown","unknown":"Unknown","other":"Other"
    }
    if s in mapping: return mapping[s]
    for k,v in mapping.items():
        if k in s: return v
    return str(x).strip().title()

def standardize_affected_anyhip(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip().lower()
    if s in {"affected","yes","1","true"}: return "Affected"
    if s in {"unaffected","no","0","false"}: return "Unaffected"
    try:
        return "Affected" if float(s) >= 0.5 else "Unaffected"
    except Exception:
        return "Affected" if s not in {"","unknown","na","none"} else "Unknown"

def standardize_stage(x):
    if pd.isna(x): return "Unknown"
    return str(x).strip().title()

def normalize_wald_raw(s: str):
    """Normalize raw waldenstrom_stage -> IIa/IIb/IIIa/IIIb (robust to casing/spacing/numerals)."""
    if pd.isna(s): return None
    t = re.sub(r'[^ivxabcd0-9]', '', str(s).lower())
    t = t.replace('3','iii').replace('2','ii')
    if t in {'iia','iib','iiia','iiib'}:
        return t.replace('iii','III').replace('ii','II')
    if t.startswith('iiib'): return 'IIIb'
    if t.startswith('iiia'): return 'IIIa'
    if t.startswith('iib'):  return 'IIb'
    if t.startswith('iia'):  return 'IIa'
    u = str(s).strip().replace(' ','').upper()
    for tok in ['IIA','IIB','IIIA','IIIB']:
        if tok in u: return tok.title().replace('Iii','III')
    return str(s).strip().title()


# ======================
# Patient ID parsing & choosing rep image
# ======================
PATIENT_REGEXES = [
    re.compile(r'(?i)\bpatient[_\-\s]*([a-z0-9]+)'),
    re.compile(r'(?i)\bpatient([0-9]+)'),
]
def extract_patient_id_from_filename(fname: str):
    if pd.isna(fname): return None
    s = str(fname).strip()
    for rx in PATIENT_REGEXES:
        m = rx.search(s)
        if m: return f"Patient_{m.group(1)}"
    low = s.lower()
    if "patient" in low:
        toks = re.split(r'[_\-\s]+', s)
        for i,t in enumerate(toks):
            if t.lower().startswith("patient"):
                return f"Patient_{toks[i+1]}" if i+1 < len(toks) else toks[i].title()
    return None

def choose_representative_row(group_df: pd.DataFrame, file_col: str):
    """Prefer 'initial', then 'ap', else lexicographic filename."""
    def score(fn):
        low = str(fn).lower()
        s1 = 1 if "initial" in low else 0
        s2 = 1 if re.search(r'(^|[_\-\s])ap([_\-\s]|$)', low) else (1 if "ap" in low else 0)
        return (-s1, -s2, str(fn))
    if file_col not in group_df.columns:
        return group_df.iloc[[0]]
    order = sorted(range(len(group_df)), key=lambda i: score(group_df.iloc[i][file_col]))
    return group_df.iloc[[order[0]]]


# ======================
# Distributions math
# ======================
def make_age_bins(age_series: pd.Series, bin_count: int = DEFAULT_AGE_BINS):
    vals = pd.to_numeric(age_series, errors="coerce").dropna()
    if len(vals) == 0: return None
    try:
        qs = np.linspace(0, 1, bin_count + 1)
        edges = sorted(set(np.quantile(vals, qs)))
        if len(edges) >= 3:
            return np.array(edges, dtype=float)
    except Exception:
        pass
    mn, mx = vals.min(), vals.max()
    if mn == mx: return np.array([mn, mx + 1e-6], dtype=float)
    return np.linspace(mn, mx, bin_count + 1, dtype=float)

def bin_age(values: np.ndarray, bins: np.ndarray):
    if bins is None or len(values) == 0: return np.zeros(0, dtype=int)
    counts, _ = np.histogram(values, bins=bins)
    return counts.astype(int)

def l1_distance_from_arrays(p_counts: np.ndarray, q_counts: np.ndarray):
    p_total = p_counts.sum(); q_total = q_counts.sum()
    if p_total == 0 and q_total == 0: return 0.0
    p = (p_counts / p_total) if p_total > 0 else np.zeros_like(p_counts, dtype=float)
    q = (q_counts / q_total) if q_total > 0 else np.zeros_like(q_counts, dtype=float)
    return float(np.abs(p - q).sum())


# ======================
# Deduplicate to one row per patient (rep image per patient)
# ======================
def dedupe_to_patients(df: pd.DataFrame, cols: dict, patient_id_col: str = None):
    work = df.copy()
    if patient_id_col:
        pid_col = coalesce_column(work, patient_id_col)
        work["PatientID"] = work[pid_col].astype(str)
    else:
        file_col = coalesce_column(work, cols["file_name"])
        work["PatientID"] = work[file_col].apply(extract_patient_id_from_filename)

    missing_pid = work["PatientID"].isna().sum()
    if missing_pid > 0:
        print(f"Warning: {missing_pid} rows missing PatientID; dropping those.")
        work = work[work["PatientID"].notna()].copy()

    reps = []; audit_rows = []
    for pid, g in work.groupby("PatientID", sort=False):
        rep = choose_representative_row(g, cols.get("file_name", "file_name")).copy()
        rep["__n_rows_for_patient"] = len(g)
        insts = g[cols["institution"]].dropna().unique().tolist() if cols.get("institution") in g.columns else []
        audit_rows.append({
            "PatientID": pid,
            "n_rows": len(g),
            "chosen_file_name": rep.iloc[0][cols.get("file_name", "file_name")] if cols.get("file_name") in rep.columns else None,
            "institutions_in_group": "|".join(map(str, insts)) if insts else "",
            "institution_conflict": len(insts) > 1
        })
        reps.append(rep)

    patients_df = pd.concat(reps, axis=0, ignore_index=True)
    audit_df = pd.DataFrame(audit_rows)

    # Standardize patients
    if cols.get("age") in patients_df.columns:
        patients_df[cols["age"]] = pd.to_numeric(patients_df[cols["age"]], errors="coerce")
    if cols.get("gender") in patients_df.columns:
        patients_df[cols["gender"]] = patients_df[cols["gender"]].apply(standardize_gender)
    if cols.get("race") in patients_df.columns:
        patients_df[cols["race"]] = patients_df[cols["race"]].apply(standardize_race_ethnicity)

    if cols.get("affected") not in patients_df.columns:
        for fallback in ["standardized_Affected_hip","Affected_hip","Bilateral"]:
            try:
                c = coalesce_column(patients_df, fallback)
                patients_df["__affected_tmp__"] = patients_df[c].apply(
                    lambda v: "Unaffected" if str(v).strip().lower() in {"unaffected","no","0","false","nan","none"} else "Affected"
                )
                cols["affected"] = "__affected_tmp__"; break
            except KeyError:
                continue
    if cols.get("affected") in patients_df.columns:
        patients_df[cols["affected"]] = patients_df[cols["affected"]].apply(standardize_affected_anyhip)

    if cols.get("stulberg") in patients_df.columns:
        patients_df[cols["stulberg"]] = patients_df[cols["stulberg"]].apply(standardize_stage)

    # RAW waldenström (patient table kept for reporting; balancing uses IMAGE-level later)
    raw_col = cols.get("waldenstrom_raw", DEFAULT_COLS["waldenstrom_raw"])
    if raw_col in patients_df.columns:
        patients_df["waldenstrom_raw_norm"] = patients_df[raw_col].apply(normalize_wald_raw)
    else:
        patients_df["waldenstrom_raw_norm"] = None

    # lateral pillar (keep; missing handled later)
    lp_col = cols.get("lateral_pillar", DEFAULT_COLS["lateral_pillar"])
    if lp_col not in patients_df.columns:
        patients_df[lp_col] = np.nan

    return patients_df, audit_df


# ======================
# Precompute per-institution stats
# (PATIENT-level for age/gender/race/affected/stulberg; IMAGE-level for waldenström RAW)
# ======================
def build_category_index(series: pd.Series):
    cats = sorted(series.dropna().unique().tolist())
    index = {c: i for i, c in enumerate(cats)}
    return cats, index

def precompute_inst_stats(patients_df: pd.DataFrame, df_all: pd.DataFrame, cols: dict, age_bins: np.ndarray):
    """
    Returns:
      inst_list, inst_stats, totals, cat_meta, age_bins_edges

    inst_stats[inst]:
      {
        # patient-level fields
        "n": <patients>, "age": np.ndarray or None,
        "gender": np.ndarray or None,
        "race": np.ndarray or None,
        "affected": np.ndarray or None,
        "stulberg": np.ndarray or None,

        # image-level fields
        "n_images": <images>,
        "waldenstrom_raw": np.ndarray or None  (IMAGE-LEVEL)
      }
    totals[...] summed likewise (note: waldenstrom_raw totals are IMAGE-LEVEL)
    """

    inst_col = cols["institution"]

    # IMAGE-level normalized Waldenström for df_all
    wraw_col_all = cols.get("waldenstrom_raw", DEFAULT_COLS["waldenstrom_raw"])
    if wraw_col_all in df_all.columns:
        df_all = df_all.copy()
        df_all["__wald_raw_norm_img__"] = df_all[wraw_col_all].apply(normalize_wald_raw)
    else:
        df_all = df_all.copy()
        df_all["__wald_raw_norm_img__"] = np.nan

    # Category metadata:
    # - patient-level categories for gender/race/affected/stulberg
    # - image-level categories for waldenstrom_raw
    cat_meta = {}

    # Patient-level categories
    for var in ["gender","race","affected","stulberg"]:
        col = cols.get(var)
        if col in patients_df.columns:
            cats, idx = build_category_index(patients_df[col])
            cat_meta[var] = {"cats": cats, "idx": idx}
        else:
            cat_meta[var] = None

    # IMAGE-level waldenström categories
    cats_w, idx_w = build_category_index(df_all["__wald_raw_norm_img__"])
    cat_meta["waldenstrom_raw"] = {"cats": cats_w, "idx": idx_w} if len(cats_w) > 0 else None

    # Institution list (union of any presence)
    inst_from_pat = set(patients_df[inst_col].dropna().unique()) if inst_col in patients_df.columns else set()
    inst_from_img = set(df_all[inst_col].dropna().unique())      if inst_col in df_all.columns else set()
    inst_list = sorted(inst_from_pat.union(inst_from_img))

    # Age bins
    age_bins_edges = age_bins if age_bins is not None else None
    age_num_bins = (len(age_bins_edges) - 1) if age_bins_edges is not None else 0

    # Totals init
    totals = {
        "n": 0,
        "n_images": 0,
        "age": np.zeros(age_num_bins, dtype=int) if age_num_bins > 0 else None,
        "gender": None, "race": None, "affected": None, "stulberg": None,
        "waldenstrom_raw": None,  # IMAGE-level totals
    }
    for var in ["gender","race","affected","stulberg"]:
        if cat_meta[var] is not None:
            totals[var] = np.zeros(len(cat_meta[var]["cats"]), dtype=int)
    if cat_meta["waldenstrom_raw"] is not None:
        totals["waldenstrom_raw"] = np.zeros(len(cat_meta["waldenstrom_raw"]["cats"]), dtype=int)

    # Pre-group
    g_pat_by_inst = {inst: g for inst, g in patients_df.groupby(inst_col)} if inst_col in patients_df.columns else {}
    g_img_by_inst = {inst: g for inst, g in df_all.groupby(inst_col)}      if inst_col in df_all.columns else {}

    inst_stats = {}

    for inst in inst_list:
        g_pat = g_pat_by_inst.get(inst)
        g_img = g_img_by_inst.get(inst)

        # Counts
        n_pat = len(g_pat) if g_pat is not None else 0
        n_img = len(g_img) if g_img is not None else 0

        # Age histogram
        if age_bins_edges is not None and g_pat is not None and cols.get("age") in g_pat.columns:
            ages = pd.to_numeric(g_pat[cols["age"]], errors="coerce").dropna().values
            age_hist = bin_age(ages, age_bins_edges)
        else:
            age_hist = None

        entry = {
            "n": int(n_pat),
            "n_images": int(n_img),
            "age": age_hist,
            "gender": None, "race": None, "affected": None, "stulberg": None,
            "waldenstrom_raw": None,
        }

        # Patient-level categoricals
        for var in ["gender","race","affected","stulberg"]:
            meta = cat_meta[var]
            if meta is None or g_pat is None:
                entry[var] = None
                continue
            col = cols.get(var)
            if col not in g_pat.columns:
                entry[var] = None
                continue
            counts = np.zeros(len(meta["cats"]), dtype=int)
            vals = g_pat[col].dropna().values
            for v in vals:
                if v in meta["idx"]:
                    counts[meta["idx"][v]] += 1
            entry[var] = counts

        # IMAGE-level waldenström
        meta_w = cat_meta["waldenstrom_raw"]
        if meta_w is not None and g_img is not None:
            counts_w = np.zeros(len(meta_w["cats"]), dtype=int)
            vals_w = g_img["__wald_raw_norm_img__"].dropna().values
            for v in vals_w:
                if v in meta_w["idx"]:
                    counts_w[meta_w["idx"][v]] += 1
            entry["waldenstrom_raw"] = counts_w

        # Save inst
        inst_stats[inst] = entry

        # Update totals
        totals["n"] += n_pat
        totals["n_images"] += n_img
        if age_hist is not None:
            totals["age"] = age_hist if totals["age"] is None else totals["age"] + age_hist
        for var in ["gender","race","affected","stulberg"]:
            arr = entry[var]
            if arr is not None:
                totals[var] = arr if totals[var] is None else totals[var] + arr
        if entry["waldenstrom_raw"] is not None:
            totals["waldenstrom_raw"] = (
                entry["waldenstrom_raw"] if totals["waldenstrom_raw"] is None
                else totals["waldenstrom_raw"] + entry["waldenstrom_raw"]
            )

    return inst_list, inst_stats, totals, cat_meta, age_bins_edges


# ======================
# Split state & scoring
# ======================
class SplitState:
    def __init__(self, inst_list, inst_stats, totals, cat_meta, age_bins_edges):
        self.test_set = set()
        self.n_total_patients = int(totals["n"])
        self.n_total_images   = int(totals["n_images"])
        self.n_total_insts    = len(inst_list)
        self.cat_meta = cat_meta
        self.age_bins_edges = age_bins_edges
        self.inst_stats = inst_stats
        self.totals = totals

        # Aggregates for Test
        self.test_n = 0
        self.test_n_images = 0
        self.test_age = np.zeros_like(totals["age"]) if totals["age"] is not None else None
        self.test_counts = {}
        for var in ["gender","race","affected","stulberg","waldenstrom_raw"]:
            arr = totals[var]
            self.test_counts[var] = np.zeros_like(arr) if arr is not None else None

    def add_inst(self, inst):
        if inst in self.test_set: return
        s = self.inst_stats[inst]
        self.test_set.add(inst)
        self.test_n += s["n"]
        self.test_n_images += s["n_images"]
        if self.test_age is not None and s["age"] is not None:
            self.test_age += s["age"]
        for var in ["gender","race","affected","stulberg","waldenstrom_raw"]:
            if self.test_counts[var] is not None and s[var] is not None:
                self.test_counts[var] += s[var]

    def remove_inst(self, inst):
        if inst not in self.test_set: return
        s = self.inst_stats[inst]
        self.test_set.remove(inst)
        self.test_n -= s["n"]
        self.test_n_images -= s["n_images"]
        if self.test_age is not None and s["age"] is not None:
            self.test_age -= s["age"]
        for var in ["gender","race","affected","stulberg","waldenstrom_raw"]:
            if self.test_counts[var] is not None and s[var] is not None:
                self.test_counts[var] -= s[var]

    def copy(self):
        c = SplitState.__new__(SplitState)
        c.test_set = set(self.test_set)
        c.n_total_patients = self.n_total_patients
        c.n_total_images   = self.n_total_images
        c.n_total_insts    = self.n_total_insts
        c.cat_meta = self.cat_meta
        c.age_bins_edges = self.age_bins_edges
        c.inst_stats = self.inst_stats
        c.totals = self.totals
        c.test_n = self.test_n
        c.test_n_images = self.test_n_images
        c.test_age = None if self.test_age is None else self.test_age.copy()
        c.test_counts = {k: (None if v is None else v.copy()) for k, v in self.test_counts.items()}
        return c


def score_state(state: SplitState, weights, target_test_frac_pat, target_test_inst_frac, target_test_img_frac):
    # size deviations
    test_frac_pat = state.test_n / state.n_total_patients if state.n_total_patients > 0 else 0.0
    test_frac_img = state.test_n_images / state.n_total_images   if state.n_total_images   > 0 else 0.0
    test_frac_inst = len(state.test_set) / state.n_total_insts   if state.n_total_insts    > 0 else 0.0
    size_dev_pat = abs(test_frac_pat - target_test_frac_pat)
    size_dev_img = abs(test_frac_img - target_test_img_frac)
    inst_dev     = abs(test_frac_inst - target_test_inst_frac)

    # L1 over distributions
    dist_score = 0.0
    per_dim = {}

    # age (patient-level)
    if state.test_age is not None and state.totals["age"] is not None:
        train_age = state.totals["age"] - state.test_age
        d = l1_distance_from_arrays(train_age, state.test_age)
        per_dim["age"] = d; dist_score += weights.get("age",1.0) * d
    else:
        per_dim["age"] = 0.0

    # patient-level categoricals
    for var in ["gender","race","affected","stulberg"]:
        test_arr = state.test_counts[var]; tot_arr = state.totals[var]
        if test_arr is None or tot_arr is None:
            per_dim[var] = 0.0; continue
        train_arr = tot_arr - test_arr
        d = l1_distance_from_arrays(train_arr, test_arr)
        per_dim[var] = d; dist_score += weights.get(var,1.0) * d

    # IMAGE-level waldenström
    test_arr = state.test_counts["waldenstrom_raw"]; tot_arr = state.totals["waldenstrom_raw"]
    if test_arr is None or tot_arr is None:
        per_dim["waldenstrom_raw"] = 0.0
    else:
        train_arr = tot_arr - test_arr
        d = l1_distance_from_arrays(train_arr, test_arr)
        per_dim["waldenstrom_raw"] = d; dist_score += weights.get("waldenstrom_raw",1.0) * d

    total_score = (
        weights["size_penalty"]  * size_dev_pat +
        weights["image_penalty"] * size_dev_img +
        weights["inst_penalty"]  * inst_dev +
        dist_score
    )

    metrics = {
        "total_score": total_score,
        "size_deviation_patients": size_dev_pat,
        "test_fraction_patients": test_frac_pat,
        "n_test_patients": state.test_n,
        "n_train_patients": state.n_total_patients - state.test_n,
        "size_deviation_images": size_dev_img,
        "test_fraction_images": test_frac_img,
        "n_test_images": state.test_n_images,
        "n_train_images": state.n_total_images - state.test_n_images,
        "inst_deviation": inst_dev,
        "test_fraction_institutions": test_frac_inst,
        "n_test_institutions": len(state.test_set),
        "n_total_institutions": state.n_total_insts,
        "per_dimension_l1": per_dim
    }
    return total_score, metrics


# ======================
# Search
# ======================
def initial_greedy(inst_list, state: SplitState, target_test_frac, size_tol,
                   weights, target_test_inst_frac, target_test_img_frac, rand):
    best_score, _ = score_state(state, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)
    remain = inst_list.copy(); rand.shuffle(remain)

    improved = True
    while improved:
        improved = False; best_delta = 0.0; best_choice = None
        for inst in remain:
            s2 = state.copy(); s2.add_inst(inst)
            frac_pat = s2.test_n        / s2.n_total_patients if s2.n_total_patients > 0 else 0.0
            frac_img = s2.test_n_images / s2.n_total_images   if s2.n_total_images   > 0 else 0.0
            if (frac_pat > target_test_frac + size_tol) or (frac_img > target_test_img_frac + size_tol):
                continue
            sc, _ = score_state(s2, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)
            delta = best_score - sc
            if delta > best_delta:
                best_delta = delta; best_choice = inst
        if best_choice is not None:
            state.add_inst(best_choice)
            remain.remove(best_choice)
            best_score, _ = score_state(state, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)
            improved = True
    return state

def local_search_refine(inst_list, state: SplitState, target_test_frac, size_tol,
                        weights, target_test_inst_frac, target_test_img_frac,
                        rand, max_iters=80, swap_samples=80):
    best_state = state.copy()
    best_score, _ = score_state(best_state, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)

    for _ in range(max_iters):
        improved = False

        # single flips
        cand_order = inst_list.copy(); rand.shuffle(cand_order)
        for inst in cand_order:
            s2 = best_state.copy()
            if inst in s2.test_set: s2.remove_inst(inst)
            else: s2.add_inst(inst)
            frac_pat = s2.test_n        / s2.n_total_patients if s2.n_total_patients > 0 else 0.0
            frac_img = s2.test_n_images / s2.n_total_images   if s2.n_total_images   > 0 else 0.0
            if (abs(frac_pat - target_test_frac) > size_tol*2.0) or (abs(frac_img - target_test_img_frac) > size_tol*2.0):
                continue
            sc, _ = score_state(s2, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)
            if sc + 1e-12 < best_score:
                best_state = s2; best_score = sc; improved = True; break
        if improved: continue

        # random swaps
        test_list = list(best_state.test_set)
        train_list = [i for i in inst_list if i not in best_state.test_set]
        if not test_list or not train_list: break

        for _ in range(swap_samples):
            a = rand.choice(test_list); b = rand.choice(train_list)
            s2 = best_state.copy(); s2.remove_inst(a); s2.add_inst(b)
            frac_pat = s2.test_n        / s2.n_total_patients if s2.n_total_patients > 0 else 0.0
            frac_img = s2.test_n_images / s2.n_total_images   if s2.n_total_images   > 0 else 0.0
            if (abs(frac_pat - target_test_frac) > size_tol*2.0) or (abs(frac_img - target_test_img_frac) > size_tol*2.0):
                continue
            sc, _ = score_state(s2, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)
            if sc + 1e-12 < best_score:
                best_state = s2; best_score = sc; improved = True; break

        if not improved: break

    return best_state, best_score

def multi_start_search(inst_list, inst_stats, totals, cat_meta, age_bins_edges,
                       target_test_frac, size_tol, target_test_inst_frac, target_test_img_frac, weights,
                       starts=20, seed=42, max_iters=80, swap_samples=80):
    rng = random.Random(seed)
    best_state = None; best_score = math.inf; best_metrics = None
    for _ in range(starts):
        rand = random.Random(rng.randint(0, 10**9))
        init = SplitState(inst_list, inst_stats, totals, cat_meta, age_bins_edges)
        init = initial_greedy(inst_list, init, target_test_frac, size_tol, weights, target_test_inst_frac, target_test_img_frac, rand)
        refined, score = local_search_refine(inst_list, init, target_test_frac, size_tol, weights, target_test_inst_frac, target_test_img_frac, rand, max_iters=max_iters, swap_samples=swap_samples)
        _, metrics = score_state(refined, weights, target_test_frac, target_test_inst_frac, target_test_img_frac)
        if score < best_score:
            best_state, best_score, best_metrics = refined, score, metrics
    return best_state, best_score, best_metrics


# ======================
# Reporting helpers
# ======================
def per_institution_summary_from_arrays(inst_list, inst_stats, cat_meta):
    rows = []
    for inst in inst_list:
        s = inst_stats[inst]
        row = {"institution": inst, "n_patients": s["n"], "n_images": s["n_images"]}
        for var, prefix in [
            ("gender","Gender"),("race","Ethnicity"),("affected","Affected"),
            ("stulberg","Stulberg"),("waldenstrom_raw","WaldenstromRaw[Images]"),
        ]:
            meta = cat_meta[var]; arr = s[var]
            if meta is None or arr is None: continue
            total = s["n"] if (var != "waldenstrom_raw") else max(s["n_images"], 1)
            for cat, idx in meta["idx"].items():
                row[f"{prefix}:{cat}"] = arr[idx] / total
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_patients", ascending=False)

def series_from_counts(arr, cats):
    if arr is None or cats is None:
        return pd.Series(dtype=float)
    total = arr.sum()
    if total == 0:
        return pd.Series([0.0]*len(cats), index=cats)
    return pd.Series(arr / total, index=cats)

def plot_compare_bars_from_arrays(train_arr, test_arr, cats, title, out_path):
    tr = series_from_counts(train_arr, cats); te = series_from_counts(test_arr, cats)
    x = np.arange(len(cats)); w = 0.4
    plt.figure(figsize=(max(6, len(cats)*0.6), 4.5))
    plt.bar(x - w/2, tr.values, w, label="Train/Val")
    plt.bar(x + w/2, te.values, w, label="Test")
    plt.title(title); plt.ylabel("Proportion")
    plt.xticks(x, [str(c) for c in cats], rotation=45, ha="right")
    ymax = max(0.01, tr.max() if len(tr)>0 else 0, te.max() if len(te)>0 else 0)
    plt.ylim(0, ymax*1.2); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def format_age_bin_labels(bins):
    labels = []
    for i in range(len(bins)-1):
        a,b = bins[i], bins[i+1]
        labels.append(f"[{a:.2f}, {b:.2f})" if i < len(bins)-2 else f"[{a:.2f}, {b:.2f}]")
    return labels


# ======================
# Main
# ======================
def main():
    ap = argparse.ArgumentParser(description="Institutional 80/20 split (patients & images) with RAW Waldenström balanced at image-level.")
    ap.add_argument("--input", required=True, help="Path to Excel (.xlsx)")
    ap.add_argument("--sheet", default=None, help="Worksheet name")
    ap.add_argument("--output", required=True, help="Output directory")

    # Targets & search
    ap.add_argument("--target-test-frac",      type=float, default=0.20, help="Target fraction of PATIENTS in test")
    ap.add_argument("--target-test-img-frac",  type=float, default=0.20, help="Target fraction of IMAGES in test")
    ap.add_argument("--target-test-inst-frac", type=float, default=0.20, help="Target fraction of INSTITUTIONS in test")
    ap.add_argument("--size-tol", type=float, default=0.02, help="Tolerance on test fractions during search")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--swap-samples", type=int, default=80)

    # Columns
    ap.add_argument("--patient-id-column", default=DEFAULT_COLS["patient_id"])
    ap.add_argument("--file-name-column",  default=DEFAULT_COLS["file_name"])
    ap.add_argument("--institution-column",default=DEFAULT_COLS["institution"])
    ap.add_argument("--age-column",        default=DEFAULT_COLS["age"])
    ap.add_argument("--gender-column",     default=DEFAULT_COLS["gender"])
    ap.add_argument("--race-column",       default=DEFAULT_COLS["race"])
    ap.add_argument("--affected-column",   default=DEFAULT_COLS["affected"])
    ap.add_argument("--stulberg-column",   default=DEFAULT_COLS["stulberg"])
    ap.add_argument("--waldenstrom-raw-column", default=DEFAULT_COLS["waldenstrom_raw"])
    ap.add_argument("--lateral-pillar-column",  default=DEFAULT_COLS["lateral_pillar"])
    ap.add_argument("--wald-subset-stages", default="IIb,IIIa",
                    help="Comma-separated raw waldenstrom stages to subset on (e.g., 'IIb,IIIa')")

    # OneDrive locks
    ap.add_argument("--copy-locked-input", type=str, default="false", help="If 'true', copy locked Excel to temp and read it")

    # Deprecated alias for standardized flag (maps to raw)
    ap.add_argument("--waldenstrom-column", dest="deprecated_wald_std", default=None,
                    help="(Deprecated) standardized Waldenström column. Use --waldenstrom-raw-column instead.")

    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Backward-compat
    if getattr(args, "deprecated_wald_std", None) and not getattr(args, "waldenstrom_raw_column", None):
        args.waldenstrom_raw_column = args.deprecated_wald_std
        print("[WARN] --waldenstrom-column is deprecated; using --waldenstrom-raw-column instead.")

    # Read Excel
    copy_locked = str(args.copy_locked_input).strip().lower() in {"1","true","yes","y"}
    if args.sheet:
        df_all = safe_read_excel(args.input, sheet_name=args.sheet, try_copy=copy_locked)
    else:
        df_all = safe_read_excel(args.input, sheet_name=None, try_copy=copy_locked)
        if isinstance(df_all, dict):
            df_all = df_all[next(iter(df_all))]
    df_all.columns = [str(c).strip() for c in df_all.columns]

    # Resolve columns (case-insensitive)
    cols = {}
    def bind(name, val):
        try:
            cols[name] = coalesce_column(df_all, val) if val is not None else None
        except KeyError as e:
            print(f"Notice: {e}"); cols[name] = None
    bind("institution", args.institution_column)
    bind("file_name", args.file_name_column)
    bind("age", args.age_column)
    bind("gender", args.gender_column)
    bind("race", args.race_column)
    bind("affected", args.affected_column)
    bind("stulberg", args.stulberg_column)
    bind("waldenstrom_raw", args.waldenstrom_raw_column)
    bind("lateral_pillar", args.lateral_pillar_column)

    # Deduplicate -> patient table (for patient-level fields)
    patients_df, audit_df = dedupe_to_patients(df_all, cols, patient_id_col=args.patient_id_column)
    total_patients = len(patients_df)
    print(f"Total unique patients: {total_patients}")

    # Age bins
    age_bins_edges = make_age_bins(patients_df[cols["age"]], DEFAULT_AGE_BINS) if cols.get("age") in patients_df.columns else None

    # Precompute per-institution (patient-level for most, IMAGE-level for waldenström)
    inst_list, inst_stats, totals, cat_meta, age_bins_edges = precompute_inst_stats(patients_df, df_all, cols, age_bins_edges)
    print(f"Found {len(inst_list)} institutions. Total images: {totals['n_images']}")

    # Summary per institution
    inst_summary = per_institution_summary_from_arrays(inst_list, inst_stats, cat_meta)
    inst_summary.to_csv(os.path.join(args.output, "institution_summary_patients.csv"), index=False)
    audit_df.to_csv(os.path.join(args.output, "patient_dedupe_audit.csv"), index=False)

    # Search best split
    best_state, best_score, best_metrics = multi_start_search(
        inst_list, inst_stats, totals, cat_meta, age_bins_edges,
        target_test_frac=args.target_test_frac,
        size_tol=args.size_tol,
        target_test_inst_frac=args.target_test_inst_frac,
        target_test_img_frac=args.target_test_img_frac,
        weights=WEIGHTS,
        starts=args.starts, seed=args.random_seed,
        max_iters=args.max_iters, swap_samples=args.swap_samples
    )

    # Patient-level split file
    test_insts = set(best_state.test_set)
    patients_split_df = patients_df[["PatientID", cols["institution"]]].copy()
    patients_split_df["split"] = patients_split_df[cols["institution"]].apply(lambda x: "Test" if x in test_insts else "Train/Val")
    patients_split_df.to_csv(os.path.join(args.output, "patients_with_split.csv"), index=False)

    # Institution assignment (patients & images)
    assign_rows = []
    for inst in inst_list:
        s = inst_stats[inst]
        assign_rows.append({"institution": inst,
                            "split": ("Test" if inst in test_insts else "Train/Val"),
                            "n_patients": s["n"],
                            "n_images": s["n_images"]})
    assign_df = pd.DataFrame(assign_rows).sort_values(["split","n_patients"], ascending=[True,False])
    assign_df.to_csv(os.path.join(args.output, "split_assignment_institutions.csv"), index=False)

    # Plots
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(args.output, f"plots_{ts}"); os.makedirs(plots_dir, exist_ok=True)

    # Age plot
    if age_bins_edges is not None and best_state.test_age is not None and totals["age"] is not None:
        labels = format_age_bin_labels(age_bins_edges)
        train_age = totals["age"] - best_state.test_age
        plot_compare_bars_from_arrays(train_age, best_state.test_age, labels,
                                      "Age Distribution (Train/Val vs Test) — Patient-Level",
                                      os.path.join(plots_dir, "age_hist.png"))

    # Categorical plots (patient-level + image-level Waldenström)
    for var, label in [
        ("gender","Gender"),
        ("race","Ethnicity"),
        ("affected","Affected vs Unaffected"),
        ("stulberg","Stulberg Stage"),
        ("waldenstrom_raw","Waldenström Stage (Raw, Image-level)"),
    ]:
        meta = cat_meta[var]
        if meta is None: continue
        cats = meta["cats"]; tot_arr = totals[var]; test_arr = best_state.test_counts[var]
        if tot_arr is None or test_arr is None: continue
        train_arr = tot_arr - test_arr
        plot_compare_bars_from_arrays(train_arr, test_arr, cats,
                                      f"{label} (Train/Val vs Test)",
                                      os.path.join(plots_dir, f"{var}.png"))

    # ==============================
    # Waldenström split analysis — IMAGE LEVEL
    # ==============================
    raw_out_csv = os.path.join(args.output, "waldenstrom_stage_raw_split_distribution.csv")
    raw_chi2 = {"stat": None, "p": None, "note": ""}
    df_all = df_all.copy()
    df_all["__wald_raw_norm_img__"] = df_all[cols["waldenstrom_raw"]].apply(normalize_wald_raw) if cols.get("waldenstrom_raw") in df_all.columns else np.nan
    df_all["__split__"] = df_all[cols["institution"]].apply(lambda x: "Test" if x in test_insts else "Train/Val")

    wr_tr = df_all.loc[df_all["__split__"]=="Train/Val", "__wald_raw_norm_img__"].value_counts().sort_index()
    wr_te = df_all.loc[df_all["__split__"]=="Test",      "__wald_raw_norm_img__"].value_counts().sort_index()
    cats_wr = sorted(set(wr_tr.index).union(set(wr_te.index)))
    tr = pd.Series(0, index=cats_wr); tr.update(wr_tr)
    te = pd.Series(0, index=cats_wr); te.update(wr_te)

    rows = []
    for label, arr in [("Train/Val", tr), ("Test", te)]:
        T = arr.sum() if arr.sum()>0 else 1
        for cat in cats_wr:
            rows.append({"split": label, "waldenstrom_stage_raw (images)": cat, "count": int(arr[cat]), "prop": float(arr[cat]/T)})
    pd.DataFrame(rows).to_csv(raw_out_csv, index=False)

    # Chi-square (image-level)
    if SCIPY_AVAILABLE and len(cats_wr)>0:
        table = np.vstack([tr.reindex(cats_wr).values, te.reindex(cats_wr).values])
        try:
            stat, p, _, _ = chi2_contingency(table)
            raw_chi2 = {"stat": float(stat), "p": float(p), "note": ""}
        except Exception:
            raw_chi2 = {"stat": None, "p": None, "note": "Chi-square failed (sparse)"}

    # ==============================
    # IIb/IIIa subset — PATIENT LEVEL (include ALL IIb/IIIa; LP kept, not filtered)
    # ==============================
    subset_csv  = os.path.join(args.output, "subset_Wald_IIb_IIIa_lateral_pillar_by_split.csv")
    subset_plot = os.path.join(plots_dir, "subset_wald_IIb_IIIa_lateral_pillar.png")
    subset_stats = {"chi2": {"stat": None, "p": None, "note": ""}, "n_train": 0, "n_test": 0}

    subset_targets = [t.strip() for t in str(args.wald_subset_stages).split(",") if t.strip()]
    subset_targets_norm = set([normalize_wald_raw(t) for t in subset_targets])

    wrn_pat = patients_df["waldenstrom_raw_norm"] if "waldenstrom_raw_norm" in patients_df.columns else pd.Series(index=patients_df.index, dtype=object)
    # lateral pillar normalize; missing -> 'Not Reported'
    lp_col = cols.get("lateral_pillar")
    if lp_col in patients_df.columns:
        lat_raw = patients_df[lp_col].astype(str).str.strip().replace({"":"nan"})
        lat_norm = lat_raw.replace({"nan": np.nan, "NaN": np.nan}).str.title().fillna("Not Reported")
    else:
        lat_norm = pd.Series("Not Reported", index=patients_df.index)

    is_test = patients_split_df["split"].values == "Test"
    mask_split = pd.Series(is_test, index=patients_df.index)

    subset_mask = wrn_pat.notna() & wrn_pat.isin(subset_targets_norm)   # <-- no LP filter
    subset_df = pd.DataFrame({
        "PatientID": patients_df.loc[subset_mask, "PatientID"],
        "waldenstrom_raw_norm": wrn_pat.loc[subset_mask],
        "lateral_pillar_norm": lat_norm.loc[subset_mask],
        "split": np.where(mask_split.loc[subset_mask], "Test", "Train/Val")
    })

    subset_stats["n_train"] = int((subset_df["split"]=="Train/Val").sum())
    subset_stats["n_test"]  = int((subset_df["split"]=="Test").sum())

    tr_lp = subset_df.loc[subset_df["split"]=="Train/Val", "lateral_pillar_norm"].value_counts().sort_index()
    te_lp = subset_df.loc[subset_df["split"]=="Test",      "lateral_pillar_norm"].value_counts().sort_index()
    cats_lp = sorted(set(tr_lp.index).union(set(te_lp.index)))
    tr2 = pd.Series(0, index=cats_lp); tr2.update(tr_lp)
    te2 = pd.Series(0, index=cats_lp); te2.update(te_lp)

    # CSV for subset
    rows_out = []
    for label, arr in [("Train/Val", tr2), ("Test", te2)]:
        T = int(arr.sum()) if arr.sum()>0 else 1
        for cat in cats_lp:
            rows_out.append({"split": label,
                             "waldenstrom_raw_subset": ",".join(sorted(subset_targets_norm)),
                             "lateral_pillar": cat,
                             "count": int(arr[cat]),
                             "prop": float(arr[cat]/T)})
    pd.DataFrame(rows_out).to_csv(subset_csv, index=False)

    # Chi-square for LP distribution in subset
    if SCIPY_AVAILABLE and len(cats_lp)>0:
        table = np.vstack([tr2.reindex(cats_lp).values, te2.reindex(cats_lp).values])
        try:
            stat, p, _, _ = chi2_contingency(table)
            subset_stats["chi2"] = {"stat": float(stat), "p": float(p), "note": ""}
        except Exception:
            subset_stats["chi2"] = {"stat": None, "p": None, "note": "Chi-square failed (sparse data)"}

    print(f"Saved plots to: {plots_dir}")

    # ==============================
    # 02_stats_tests — REINSTATED
    # ==============================
    stats_tests = {}

    # Patient-level AGE KS
    if cols.get("age") in patients_df.columns:
        is_test_pat = patients_split_df["split"].values == "Test"
        mask_pat = pd.Series(is_test_pat, index=patients_df.index)
        tr_age = patients_df.loc[~mask_pat, cols["age"]]
        te_age = patients_df.loc[mask_pat, cols["age"]]
        if SCIPY_AVAILABLE:
            a = pd.to_numeric(tr_age, errors="coerce").dropna()
            b = pd.to_numeric(te_age, errors="coerce").dropna()
            if len(a) and len(b):
                stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
                stats_tests["age_ks"] = {"stat": float(stat), "p": float(p), "note": ""}
            else:
                stats_tests["age_ks"] = {"stat": None, "p": None, "note": "Insufficient age data"}
        else:
            stats_tests["age_ks"] = {"stat": None, "p": None, "note": "SciPy not installed"}

    # Helper: chi2 on patient-level categorical columns
    def chi2_on_patients(colname: str, label: str):
        if colname not in patients_df.columns:
            return { "stat": None, "p": None, "note": "Column missing" }
        is_test_pat = patients_split_df["split"].values == "Test"
        mask_pat = pd.Series(is_test_pat, index=patients_df.index)
        a = patients_df.loc[~mask_pat, colname].astype(str)
        b = patients_df.loc[mask_pat, colname].astype(str)
        if not SCIPY_AVAILABLE:
            return {"stat": None, "p": None, "note": "SciPy not installed"}
        cats = sorted(set(a.unique()).union(set(b.unique())))
        a_counts = pd.Series(0, index=cats); a_counts.update(a.value_counts())
        b_counts = pd.Series(0, index=cats); b_counts.update(b.value_counts())
        try:
            stat, p, _, _ = chi2_contingency(np.vstack([a_counts.values, b_counts.values]))
            return {"stat": float(stat), "p": float(p), "note": ""}
        except Exception:
            return {"stat": None, "p": None, "note": "Chi-square failed (sparse)"}

    # Patient-level χ² for gender/race/affected/stulberg
    for var in ["gender","race","affected","stulberg"]:
        col = cols.get(var)
        if col:
            stats_tests[f"{var}_chi2"] = chi2_on_patients(col, var)

    # Image-level Waldenström χ²
    stats_tests["waldenstrom_raw_images_chi2"] = {
        "stat": raw_chi2.get("stat"),
        "p":    raw_chi2.get("p"),
        "note": raw_chi2.get("note", "")
    }

    # IIb/IIIa subset LP χ² (patient-level)
    stats_tests["wald_iib_iiia_lateral_pillar_chi2"] = {
        "stat": subset_stats["chi2"].get("stat"),
        "p":    subset_stats["chi2"].get("p"),
        "note": subset_stats["chi2"].get("note","")
    }

    # ==============
    # Excel report
    # ==============
    xlsx_path = os.path.join(args.output, "split_summary_patients.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # 00) Summary
        pd.DataFrame({
            "metric": [
                "total_unique_patients",
                "train_val_patients_n",
                "test_patients_n",
                "test_fraction_patients",
                "target_test_fraction_patients",
                "size_deviation_patients",

                "train_val_images_n",
                "test_images_n",
                "test_fraction_images",
                "target_test_fraction_images",
                "size_deviation_images",

                "test_institutions_n",
                "total_institutions_n",
                "test_fraction_institutions",
                "target_test_fraction_institutions",
                "institution_fraction_deviation",

                "objective_score",
            ],
            "value": [
                total_patients,
                best_metrics["n_train_patients"],
                best_metrics["n_test_patients"],
                best_metrics["test_fraction_patients"],
                args.target_test_frac,
                best_metrics["size_deviation_patients"],

                best_metrics["n_train_images"],
                best_metrics["n_test_images"],
                best_metrics["test_fraction_images"],
                args.target_test_img_frac,
                best_metrics["size_deviation_images"],

                best_metrics["n_test_institutions"],
                best_metrics["n_total_institutions"],
                best_metrics["test_fraction_institutions"],
                args.target_test_inst_frac,
                best_metrics["inst_deviation"],

                best_metrics["total_score"],
            ]
        }).to_excel(writer, index=False, sheet_name="00_summary")

        # 01) L1 distances
        pd.DataFrame(
            [{"dimension": k, "l1_distance": v} for k,v in best_metrics["per_dimension_l1"].items()]
        ).to_excel(writer, index=False, sheet_name="01_l1_distances")

        # 02) STATS TESTS (reinstated)
        pd.DataFrame(
            [{"test": k, "stat": v.get("stat"), "p": v.get("p"), "note": v.get("note","")}
             for k,v in stats_tests.items()]
        ).to_excel(writer, index=False, sheet_name="02_stats_tests")

        # 03) Institution summary
        per_institution_summary_from_arrays(inst_list, inst_stats, cat_meta).to_excel(writer, index=False, sheet_name="03_institution_summary")

        # 04) Distributions table
        # Note: waldenstrom_raw here is IMAGE-LEVEL; label it explicitly
        dist_rows = []
        # Age (patient)
        if age_bins_edges is not None and totals["age"] is not None and best_state.test_age is not None:
            train_age = totals["age"] - best_state.test_age
            labels_age = format_age_bin_labels(age_bins_edges)
            for label, arr in [("Train/Val", train_age), ("Test", best_state.test_age)]:
                T = arr.sum() if arr.sum()>0 else 1
                for cat_label, cnt in zip(labels_age, arr):
                    dist_rows.append({"variable": "age (patients)", "category": cat_label,
                                      "count": int(cnt), "prop": float(cnt/T), "split": label})

        # Patient-level categoricals
        for var, nice in [("gender","gender (patients)"),
                          ("race","ethnicity (patients)"),
                          ("affected","affected (patients)"),
                          ("stulberg","stulberg (patients)")]:
            meta = cat_meta[var]
            if meta is None: continue
            cats = meta["cats"]; tot_arr = totals[var]; test_arr = best_state.test_counts[var]
            if tot_arr is None or test_arr is None: continue
            train_arr = tot_arr - test_arr
            for label, arr in [("Train/Val", train_arr), ("Test", test_arr)]:
                T = arr.sum() if arr.sum()>0 else 1
                for cat, cnt in zip(cats, arr):
                    dist_rows.append({"variable": nice, "category": cat, "count": int(cnt), "prop": float(cnt/T), "split": label})

        # IMAGE-level waldenström
        meta_w = cat_meta["waldenstrom_raw"]
        if meta_w is not None:
            cats = meta_w["cats"]; tot_arr = totals["waldenstrom_raw"]; test_arr = best_state.test_counts["waldenstrom_raw"]
            if tot_arr is not None and test_arr is not None:
                train_arr = tot_arr - test_arr
                for label, arr in [("Train/Val", train_arr), ("Test", test_arr)]:
                    T = arr.sum() if arr.sum()>0 else 1
                    for cat, cnt in zip(cats, arr):
                        dist_rows.append({"variable": "waldenstrom_raw (images)", "category": cat,
                                          "count": int(cnt), "prop": float(cnt/T), "split": label})
        pd.DataFrame(dist_rows).to_excel(writer, index=False, sheet_name="04_distributions")

        # 05) Assignments & 06) Patients & 07) Audit
        assign_df.to_excel(writer, index=False, sheet_name="05_split_assignment_institutions")
        patients_split_df.to_excel(writer, index=False, sheet_name="06_patients_with_split")
        audit_df.to_excel(writer, index=False, sheet_name="07_patient_dedupe_audit")

        # 08) Waldenström split (images)
        pd.read_csv(raw_out_csv).to_excel(writer, index=False, sheet_name="08_waldenstrom_raw_images")

        # 09) IIb/IIIa subset (patients)
        if os.path.exists(subset_csv):
            df_sub = pd.read_csv(subset_csv)
            extra = pd.DataFrame([{
                "split": "STATS",
                "waldenstrom_raw_subset": ",".join(sorted(subset_targets_norm)),
                "lateral_pillar": "chi2",
                "count": subset_stats["chi2"]["stat"],
                "prop": subset_stats["chi2"]["p"]
            }])
            pd.concat([df_sub, extra], ignore_index=True).to_excel(writer, index=False,
                                                                   sheet_name="09_subset_Wald_IIb_IIIa_with_LP")
        else:
            pd.DataFrame({"note": ["Subset analysis not available (missing raw waldenstrom_stage)"]}).to_excel(writer, index=False, sheet_name="09_subset_Wald_IIb_IIIa_with_LP")

    # JSON summary
    with open(os.path.join(args.output, "split_result_summary.json"), "w") as f:
        json.dump({
            "test_institutions": sorted(list(test_insts)),
            "metrics": best_metrics,
            "scipy_available": SCIPY_AVAILABLE,
            "timestamp": ts
        }, f, indent=2)

    print(f"\nSaved outputs in: {args.output}")
    print("Done.")

if __name__ == "__main__":
    main()