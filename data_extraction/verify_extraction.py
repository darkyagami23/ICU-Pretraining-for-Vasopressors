#!/usr/bin/env python3
"""
verify_extraction.py
====================

Checks whether the recovered extraction scripts
(vasopressor_data_extraction_v2.py, control_data_extraction_v3.py,
vasopressor_label_extraction.py) reproduce the stored training files
that the paper's models were trained on.

It answers seven questions (see the instructions document):

  Q1  Does each positive window END or BEGIN h minutes before infusion start?
  Q2  Do the recovered filters reproduce the stored ABP and ECG windows?
  Q3  How was positive PPG normalised (the class-dependent step)?
  Q4  Which 30-s segment were the CONTROL blood-pressure targets taken from?
  Q5  Are the POSITIVE blood-pressure targets the last 30 s before infusion?
  Q6  Does the recovered dose rule give the same infusion starts?
  Q7  Where in surgery were the control segments placed?

Nothing is modified. All results go to the --out folder.

Usage (see the instructions for details):

  python verify_extraction.py \
      --stored-dir  PATH/Master_Training_Data_Expanded_New_2 \
      --bp-target-dir PATH/BP_target \
      --vital-dir   PATH/vital_files          (omit to download from VitalDB) \
      --clinical-csv PATH/clinical_data.csv   (optional) \
      --out verify_results

Requires: numpy, pandas, scipy, vitaldb, (tqdm optional).
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, fftconvolve, resample_poly

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kw):
        return x

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ----------------------------------------------------------------------
# Constants copied from the recovered scripts
# ----------------------------------------------------------------------
FS_RAW = 500
FS = 125
WIN = 3750                       # samples in a 30-s window at 125 Hz
SEG_SEC = 30
BLOCK_SEC = 900                  # 15-min block
OFFSET_SEC = 30                  # block ends 30 s before infusion start (v2)
GAP_LIMIT_SAMPLES_RAW = int(1.0 * FS_RAW)

WAVE_TRACKS = {"ppg": "SNUADC/PLETH", "ecg": "SNUADC/ECG_II", "abp": "SNUADC/ART"}
WAVE_ORDER = ["ppg", "ecg", "abp"]
NUM_TRACKS = {"sbp": "Solar8000/ART_SBP", "dbp": "Solar8000/ART_DBP",
              "map": "Solar8000/ART_MBP"}
NUM_ORDER = ["map", "sbp", "dbp"]
DRUG_KEYS = ["PHEN_RATE", "NEPI_RATE", "EPI_RATE", "VASO_RATE", "DOPA_RATE"]
DRUG_TRACKS = ["Orchestra/" + k for k in DRUG_KEYS]
MIN_RATE_BY_DRUG = {"NEPI_RATE": 1.0, "EPI_RATE": 1.0, "PHEN_RATE": 1.0,
                    "DOPA_RATE": 1.0, "VASO_RATE": 0.5}
GAP_MERGE_SEC = 60
MIN_EVENT_SEC = 10
IDX15 = np.linspace(0, SEG_SEC - 1, 15).astype(int)   # extract_15_points on 30 samples

TOL_SEC = 2.0                    # tolerance for "same time" decisions


# ----------------------------------------------------------------------
# Signal processing: identical to vasopressor_data_extraction_v2.py
# ----------------------------------------------------------------------
def bandpass(x, lo, hi, fs=FS_RAW, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def lowpass(x, cutoff=25.0, fs=FS_RAW, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def fill_gaps_and_zeropad(x):
    s = pd.Series(np.asarray(x, dtype=float))
    s = s.interpolate(method="pchip", limit=GAP_LIMIT_SAMPLES_RAW,
                      limit_direction="forward")
    return s.fillna(0.0).values


def process_v2(raw, mod):
    """One modality, 500 Hz raw -> 125 Hz, exactly as the v2 script."""
    if mod == "ppg":
        y = bandpass(fill_gaps_and_zeropad(raw), 0.5, 25.0)
    elif mod == "ecg":
        y = bandpass(fill_gaps_and_zeropad(np.clip(raw, -5.0, 5.0)), 0.5, 40.0)
    else:
        y = lowpass(fill_gaps_and_zeropad(np.clip(raw, 0.0, 350.0)), 25.0)
    return resample_poly(y, FS, FS_RAW)


def extract_15_points(seg):
    clean = seg[~np.isnan(seg)]
    if len(clean) < 2:
        return np.full(15, np.nan)
    idx = np.linspace(0, len(clean) - 1, 15).astype(int)
    return clean[idx]


def detect_events(values, min_rate, gap_sec=GAP_MERGE_SEC, min_dur=MIN_EVENT_SEC):
    """Same as vasopressor_label_extraction.py."""
    active = np.where((values > min_rate) & (~np.isnan(values)))[0]
    if active.size == 0:
        return []
    breaks = np.where(np.diff(active) > gap_sec)[0]
    starts = np.insert(active[breaks + 1], 0, active[0])
    ends = np.append(active[breaks], active[-1])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_dur]


def rule_free_onsets(values):
    """Every time a pump rate goes from 0/missing to > 0 (no threshold, no merge)."""
    on = np.nan_to_num(values, nan=0.0) > 0
    prev = np.concatenate([[False], on[:-1]])
    return list(np.where(on & ~prev)[0].astype(int))


# ----------------------------------------------------------------------
# Matching helpers
# ----------------------------------------------------------------------
TRIM = 375                       # ignore 3 s at each end of a stored window when locating
EDGE_EXCL = 10 * FS              # never match within 10 s of the record start or end


def ncc_curve(y, t):
    """Normalised cross-correlation of template t at every position of y."""
    n = len(t)
    if len(y) < n:
        return None
    t0 = t - np.mean(t)
    st = np.sqrt(np.sum(t0 ** 2))
    if not np.isfinite(st) or st == 0:
        return None
    y = y - np.mean(y)
    num = fftconvolve(y, t0[::-1], mode="valid")
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    var = s2 - s1 * s1 / n
    var[var < 1e-9] = np.nan
    r = num / (np.sqrt(var) * st)
    r[:EDGE_EXCL] = np.nan
    r[-EDGE_EXCL:] = np.nan
    return r if np.any(np.isfinite(r)) else None


def window_curve(y, stored):
    """NCC curve indexed by the START sample of the full 3,750-sample window.
    Uses the middle of the window only, so block-edge filter effects do not bias it."""
    t = np.nan_to_num(stored[TRIM:WIN - TRIM])
    r = ncc_curve(y, t)
    if r is None:
        return None
    out = np.full(len(y), np.nan)
    idx = np.arange(len(r)) - TRIM                 # window start = template start - TRIM
    ok = idx >= 0
    out[idx[ok]] = r[ok]
    return out


def peak(r):
    k = int(np.nanargmax(r))
    r1 = float(r[k])
    rm = r.copy()
    rm[max(0, k - 2 * FS): k + 2 * FS + 1] = np.nan  # exclude +/- 2 s
    r2 = float(np.nanmax(rm)) if np.any(np.isfinite(rm)) else np.nan
    return k, r1, r2


def linear_fit(x, s):
    """Fit s ~ a*x + b. Returns a, b, R^2, residual RMS."""
    ok = np.isfinite(x) & np.isfinite(s)
    x, s = x[ok], s[ok]
    if len(x) < 10 or np.std(x) == 0 or np.std(s) == 0:
        return np.nan, np.nan, np.nan, np.nan
    a = np.cov(x, s, bias=True)[0, 1] / np.var(x)
    b = np.mean(s) - a * np.mean(x)
    res = s - (a * x + b)
    r2 = 1.0 - np.var(res) / np.var(s)
    return float(a), float(b), float(r2), float(np.sqrt(np.mean(res ** 2)))


# ----------------------------------------------------------------------
# Stored files
# ----------------------------------------------------------------------
def read_stored(path, want_values=True):
    """Return (meta dict of the first 6 columns, header, values array)."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        row = next(r)
    meta = dict(zip(header[:6], row[:6]))
    vals = None
    if want_values:
        vals = np.array([float(v) if v not in ("", "nan", "NaN") else np.nan
                         for v in row[6:]], dtype=float)[:WIN]
    return meta, header, vals


def build_inventory(stored_dir, read_meta=True):
    pattern = os.path.join(stored_dir, "*", "abp", "offset_*m", "*_wave.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            f"No stored windows found with pattern\n  {pattern}\n"
            "Check --stored-dir: it must contain train/, val/, test/, each with "
            "abp/ ppg/ ecg/ and offset_3m ... offset_15m folders.")
    rows = []
    for p in tqdm(paths, desc="inventory"):
        parts = os.path.normpath(p).split(os.sep)
        split, window, fname = parts[-4], parts[-2], parts[-1]
        h = float(window.replace("offset_", "").replace("m", ""))
        m = re.search(r"Case_(\d+)", fname)
        d = re.search(r"dose_(\d+)", fname)
        stem = fname.replace("_abp_wave.csv", "")
        ci = stem.find("Case_")
        rec = dict(split=split, window=window, horizon=h, fname=fname, path=p,
                   case_id=int(m.group(1)) if m else -1,
                   dose=int(d.group(1)) if d else -1,
                   case_stem=stem[ci:] if ci >= 0 else stem,
                   is_control=("control" in fname.lower()))
        if read_meta:
            try:
                meta, header, _ = read_stored(p, want_values=False)
                for k, v in meta.items():
                    rec["meta_" + k] = v
            except Exception as e:  # unreadable file
                rec["meta_error"] = str(e)[:80]
        rows.append(rec)
    inv = pd.DataFrame(rows)
    if "meta_label_binary" in inv.columns:
        lb = pd.to_numeric(inv["meta_label_binary"], errors="coerce")
        inv["label"] = np.where(lb.notna(), lb, (~inv.is_control).astype(int)).astype(int)
    else:
        inv["label"] = (~inv.is_control).astype(int)
    return inv


def sibling_path(abp_path, fname, mod):
    """Path of the PPG/ECG file that belongs to an ABP file (downstream naming)."""
    window_dir = os.path.dirname(abp_path)
    split_dir = os.path.dirname(os.path.dirname(window_dir))
    window = os.path.basename(window_dir)
    return os.path.join(split_dir, mod, window, fname.replace("abp", mod))


def load_bp_target(bp_dir, kind, case_stem):
    path = os.path.join(bp_dir, f"{kind}_target", f"{case_stem}_{kind}.csv")
    if not os.path.exists(path):
        return None, {}, path
    df = pd.read_csv(path)
    if df.empty:
        return None, {}, path
    cols = sorted([c for c in df.columns
                   if c.startswith(f"{kind}_") and c.rsplit("_", 1)[-1].isdigit()],
                  key=lambda c: int(c.rsplit("_", 1)[-1]))[:15]
    meta = {c: df.iloc[0][c] for c in df.columns[:6] if c not in cols}
    vals = pd.to_numeric(df.iloc[0][cols], errors="coerce").values.astype(float)
    return vals, meta, path


# ----------------------------------------------------------------------
# VitalDB access
# ----------------------------------------------------------------------
_VITAL_INDEX = None


def find_vital(vital_dir, case_id):
    global _VITAL_INDEX
    if _VITAL_INDEX is None:
        _VITAL_INDEX = {}
        for f in glob.glob(os.path.join(vital_dir, "**", "*.vital"), recursive=True):
            m = re.search(r"(\d+)", os.path.basename(f))
            if m:
                _VITAL_INDEX[int(m.group(1))] = f
    return _VITAL_INDEX.get(case_id)


def open_case(case_id, vital_dir, tracks):
    """One VitalFile object per case, so every track shares the same time origin."""
    import vitaldb
    if vital_dir:
        path = find_vital(vital_dir, case_id)
        if path is None:
            raise FileNotFoundError(f"no .vital file for case {case_id} in {vital_dir}")
        return vitaldb.VitalFile(path, tracks)
    return vitaldb.VitalFile(case_id, tracks)      # downloads from the open dataset


def case_arrays(vf):
    names = set(vf.get_track_names())
    missing = [t for t in WAVE_TRACKS.values() if t not in names]
    if missing:
        raise ValueError(f"missing waveform tracks {missing}")
    waves = vf.to_numpy([WAVE_TRACKS[m] for m in WAVE_ORDER], interval=1 / FS_RAW)
    num_tr = [NUM_TRACKS[k] for k in NUM_ORDER]
    num = vf.to_numpy(num_tr, interval=1) if any(t in names for t in num_tr) else None
    drugs_present = [t for t in DRUG_TRACKS if t in names]
    drugs = vf.to_numpy(drugs_present, interval=1) if drugs_present else None
    return waves.astype(float), (None if num is None else num.astype(float)), \
        drugs_present, (None if drugs is None else drugs.astype(float))


def case_events(drugs_present, drugs):
    rule, free = [], []
    if drugs is None:
        return rule, free
    for j, t in enumerate(drugs_present):
        key = t.split("/")[-1]
        v = drugs[:, j]
        for s, _ in detect_events(v, MIN_RATE_BY_DRUG.get(key, 1.0)):
            rule.append((s, key))
        for s in rule_free_onsets(v):
            free.append((s, key))
    return sorted(rule), sorted(free)


# ----------------------------------------------------------------------
# Per-case check
# ----------------------------------------------------------------------
def check_case(case_stem, group, args, clinical):
    """group: inventory rows for one case_stem (all horizons)."""
    case_id = int(group.case_id.iloc[0])
    is_ctrl = bool(group.is_control.iloc[0])
    tracks = list(WAVE_TRACKS.values()) + list(NUM_TRACKS.values()) + DRUG_TRACKS
    vf = open_case(case_id, args.vital_dir, tracks)
    waves, num, drugs_present, drugs = case_arrays(vf)
    rec_len_sec = len(waves) / FS_RAW

    # whole-record processing, used only to LOCATE each stored window
    full = {m: process_v2(waves[:, i], m) for i, m in enumerate(WAVE_ORDER)}
    rule_ev, free_ev = case_events(drugs_present, drugs)

    # ---- locate every stored window (individually, then jointly) ----
    rows_sorted = [r for _, r in group.sort_values("horizon").iterrows()]
    stored_all, curves = {}, {}
    for row in rows_sorted:
        h = float(row.horizon)
        st = {}
        for m in ["abp", "ecg", "ppg"]:
            p = row.path if m == "abp" else sibling_path(row.path, row.fname, m)
            try:
                _, _, st[m] = read_stored(p)
            except Exception:
                st[m] = None
        stored_all[h] = st
        for m in ["abp", "ecg"]:                   # ABP first; ECG if ABP is flat
            if st.get(m) is None:
                continue
            c = window_curve(full[m], st[m])
            if c is not None:
                curves[h] = (m, c)
                break

    # joint search: if all windows come from one 15-min block, then
    # block_end ('ends' convention) = window_start + 60*h for every horizon
    joint_be = None
    if len(curves) >= 2:
        n = len(full["abp"])
        acc = np.zeros(n + int(60 * max(curves) * FS) + 1)
        cnt = np.zeros_like(acc)
        for h, (_, c) in curves.items():
            sh = int(round(60 * h * FS))
            v = np.nan_to_num(c, nan=0.0)
            acc[sh:sh + n] += v
            cnt[sh:sh + n] += np.isfinite(c)
        score = np.where(cnt == len(curves), acc / np.maximum(cnt, 1), np.nan)
        if np.any(np.isfinite(score)):
            joint_be = int(np.nanargmax(score))

    win_rows = []
    located = {}                      # horizon -> window start (s) used for the checks
    n_agree = 0
    indiv = {}
    for h, (m, c) in curves.items():
        k, r1, r2 = peak(c)
        indiv[h] = (m, k, r1, r2)
        if joint_be is not None and abs(k - (joint_be - int(round(60 * h * FS)))) <= TOL_SEC * FS:
            n_agree += 1
    use_joint = joint_be is not None and n_agree >= max(2, len(curves) / 2)

    for row in rows_sorted:
        h = float(row.horizon)
        stored = stored_all[h]
        if h not in indiv:
            win_rows.append(dict(case_stem=case_stem, case_id=case_id, is_control=is_ctrl,
                                 split=row.split, horizon=h,
                                 note="could not locate (flat or missing)"))
            continue
        loc_mod, k_ind, r1, r2 = indiv[h]
        k_joint = joint_be - int(round(60 * h * FS)) if joint_be is not None else None
        k = k_joint if use_joint else k_ind
        ws = k / FS
        we = ws + SEG_SEC
        located[h] = ws
        base = dict(case_stem=case_stem, case_id=case_id, is_control=is_ctrl,
                    split=row.split, horizon=h, locate_modality=loc_mod,
                    locate_r=r1, locate_r_second=r2,
                    individual_start_sec=k_ind / FS,
                    joint_start_sec=(k_joint / FS if k_joint is not None else np.nan),
                    individual_agrees_with_joint=(k_joint is not None and
                                                  abs(k_ind - k_joint) <= TOL_SEC * FS),
                    used_joint=use_joint)
        base.update(window_start_sec=ws, window_end_sec=we,
                    implied_block_end_A=we + 60 * h - OFFSET_SEC,   # 'ends h min before'
                    implied_block_end_B=we + 60 * h - 2 * OFFSET_SEC)  # 'begins h min before'

        # Q1/Q6: nearest pump events to the onset implied by 'ends h min before'
        implied_onset = we + 60 * h
        for tag, evs in (("rule", rule_ev), ("free", free_ev)):
            if evs:
                d = [e[0] - implied_onset for e in evs]
                j = int(np.argmin(np.abs(d)))
                base[f"onset_delta_{tag}_sec"] = float(d[j])
                base[f"onset_drug_{tag}"] = evs[j][1]
                base[f"onset_{tag}_sec"] = float(evs[j][0])
            else:
                base[f"onset_delta_{tag}_sec"] = np.nan

        # Q2/Q3: reproduce the block exactly as v2 under both conventions
        for hyp in ("A", "B"):
            be = base[f"implied_block_end_{hyp}"]
            bs = be - BLOCK_SEC
            i0, i1 = int(round(bs * FS_RAW)), int(round(be * FS_RAW))
            if i0 < 0 or i1 > len(waves):
                continue
            blk = {m: process_v2(waves[i0:i1, i], m) for i, m in enumerate(WAVE_ORDER)}
            s0 = int(round((ws - bs) * FS))
            for m in ["abp", "ecg", "ppg"]:
                sv = stored.get(m)
                if sv is None or s0 < 0 or s0 + WIN > len(blk[m]):
                    continue
                x = blk[m][s0:s0 + WIN]
                a, b, r2fit, rms = linear_fit(x, sv)
                base[f"{m}_R2_{hyp}"] = r2fit
                if hyp == "A":
                    base[f"{m}_resid_rms_A"] = rms
                    base[f"{m}_a_x_sdwin"] = a * np.std(x) if np.isfinite(a) else np.nan
                    base[f"{m}_a_x_sdblock"] = a * np.std(blk[m]) if np.isfinite(a) else np.nan
                    base[f"{m}_offset_win"] = b + a * np.mean(x) if np.isfinite(a) else np.nan
                    base[f"{m}_offset_block"] = b + a * np.mean(blk[m]) if np.isfinite(a) else np.nan
                    base[f"{m}_stored_mean"] = float(np.nanmean(sv))
                    base[f"{m}_stored_sd"] = float(np.nanstd(sv))
        win_rows.append(base)

    # Q4/Q5: which 30-s segment the BP target matches
    bp = dict(case_stem=case_stem, case_id=case_id, is_control=is_ctrl,
              record_len_sec=rec_len_sec)
    targets = {}
    for kind in NUM_ORDER:
        vals, meta, path = load_bp_target(args.bp_target_dir, kind, case_stem)
        targets[kind] = vals
        for mk, mv in meta.items():
            bp[f"target_meta_{kind}_{mk}"] = mv
    bp["target_files_found"] = sum(v is not None for v in targets.values())
    if num is not None and bp["target_files_found"] == 3:
        T = num.shape[0]
        best = {}
        for style in ("ffill_all", "ffill_60"):
            df = pd.DataFrame(num, columns=NUM_ORDER)
            if style == "ffill_all":
                df = df.ffill().bfill()
            else:
                df = df.ffill(limit=60).bfill(limit=60)
            arr = df.values
            n_start = T - SEG_SEC + 1
            if n_start <= 0:
                continue
            if not np.isnan(arr).any():
                idx = np.arange(n_start)[:, None] + IDX15[None, :]
                cand = arr[idx]                                    # (n_start, 15, 3)
            else:
                cand = np.stack([np.stack([extract_15_points(arr[s:s + SEG_SEC, j])
                                           for j in range(3)], axis=1)
                                 for s in range(n_start)])
            tgt = np.stack([targets[k] for k in NUM_ORDER], axis=1)  # (15, 3)
            diff = np.nanmax(np.abs(cand - tgt[None]), axis=(1, 2))
            diff[np.all(np.isnan(cand), axis=(1, 2))] = np.inf
            s_best = int(np.argmin(diff))
            best[style] = (s_best, float(diff[s_best]),
                           int(np.sum(diff < 0.01)))
        if best:
            style = min(best, key=lambda k: best[k][1])
            s_best, dmin, n_exact = best[style]
            bp.update(target_seg_start_sec=s_best, target_max_abs_diff=dmin,
                      target_n_exact_matches=n_exact, target_fill_style=style)
            if located:
                # control reference: end of the 15-min block under each convention
                hmax = max(located)
                block_end_A = located[hmax] + SEG_SEC + 60 * hmax - OFFSET_SEC
                bp["block_end_A_sec"] = block_end_A
                bp["target_start_minus_block_end_A"] = s_best - block_end_A
                bp["target_start_minus_block_end_B"] = s_best - (block_end_A - OFFSET_SEC)
            if not is_ctrl:
                # positive reference: the actual pump onset matched to the windows
                ons = [r.get("onset_free_sec") for r in win_rows if r.get("onset_free_sec") is not None]
                ons = [o for o in ons if np.isfinite(o)]
                if ons:
                    bp["pump_onset_sec"] = float(np.median(ons))
                    bp["target_start_minus_onset"] = s_best - float(np.median(ons))

    # Q7: placement of the control block
    if located:
        hmax = max(located)
        block_end_A = located[hmax] + SEG_SEC + 60 * hmax - OFFSET_SEC
        bp["block_end_frac_of_record"] = (block_end_A + OFFSET_SEC) / rec_len_sec
        if clinical is not None and case_id in clinical.index:
            c = clinical.loc[case_id]
            try:
                dur = float(c["caseend"]) - float(c["casestart"])
                bp["block_frac_of_case_v3def"] = (block_end_A + OFFSET_SEC) / dur
                mid = block_end_A - BLOCK_SEC / 2
                bp["block_mid_frac_of_op"] = (mid - float(c["opstart"])) / \
                    (float(c["opend"]) - float(c["opstart"]))
            except Exception:
                pass
        ends = [indiv[h][1] / FS + SEG_SEC + 60 * h - OFFSET_SEC for h in indiv]
        bp["block_end_spread_across_horizons_sec"] = float(np.ptp(ends)) if len(ends) > 1 else 0.0
        bp["located_with_joint_search"] = use_joint

    ev = dict(case_stem=case_stem, case_id=case_id, is_control=is_ctrl,
              n_rule_events=len(rule_ev), n_free_onsets=len(free_ev),
              drug_tracks=";".join(t.split("/")[-1] for t in drugs_present))
    return win_rows, bp, ev


# ----------------------------------------------------------------------
# All-case event count (Q6, optional)
# ----------------------------------------------------------------------
def count_all_events(inv, args):
    pos = inv[~inv.is_control]
    stored_doses = pos.groupby("case_id").dose.nunique()
    rows = []
    for cid in tqdm(sorted(stored_doses.index), desc="event count (all positives)"):
        try:
            vf = open_case(int(cid), args.vital_dir, DRUG_TRACKS)
            names = set(vf.get_track_names())
            present = [t for t in DRUG_TRACKS if t in names]
            drugs = vf.to_numpy(present, interval=1).astype(float) if present else None
            rule, free = case_events(present, drugs)
            rows.append(dict(case_id=cid, stored_doses=int(stored_doses[cid]),
                             rule_events=len(rule), free_onsets=len(free)))
        except Exception as e:
            rows.append(dict(case_id=cid, stored_doses=int(stored_doses[cid]),
                             error=str(e)[:100]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
def pct(x):
    return f"{100 * x:.0f}%"


def summarise(inv, win, bp, evc, allev, out):
    L = []
    w = L.append
    w("=" * 72)
    w("  VERIFICATION SUMMARY")
    w("=" * 72)

    # Inventory
    w("\n[0] Stored data inventory (windows per split and horizon; positives)")
    tab = inv.groupby(["split", "horizon"]).label.agg(["count", "sum"])
    w(tab.rename(columns={"count": "windows", "sum": "positives"}).to_string())
    w(f"  Positive cases: {inv[~inv.is_control].case_id.nunique()}   "
      f"positive (case, dose) pairs: {inv[~inv.is_control].case_stem.nunique()}   "
      f"control cases: {inv[inv.is_control].case_id.nunique()}")
    w("  Paper expects: 112 positive cases, 143 infusion starts, 2,134 controls, "
      "test 232 windows / 18 positives per horizon.")
    if "meta_time_offset" in inv.columns:
        w("\n  'time_offset' label stored in the files (most common values):")
        for (ctrl, h), g in inv.groupby(["is_control", "horizon"]):
            vc = g.meta_time_offset.value_counts().head(3)
            w(f"    {'control ' if ctrl else 'positive'} {h:>4.0f} min: " +
              ", ".join(f"{k} (n={v})" for k, v in vc.items()))

    if win.empty:
        w("\nNo windows were checked (see errors.csv).")
        open(os.path.join(out, "summary.txt"), "w").write("\n".join(L))
        print("\n".join(L))
        return

    pos = win[~win.is_control]
    ctl = win[win.is_control]

    if "individual_agrees_with_joint" in win.columns:
        ag = win.individual_agrees_with_joint.dropna().astype(bool)
        lr = win.locate_r.dropna()
        w(f"\n[Location] windows found: {len(lr)}; median match r {lr.median():.4f}; "
          f"windows whose own best match agrees with the joint (same-block) position: "
          f"{pct(ag.mean()) if len(ag) else 'n/a'}")
        low = (lr < 0.95).mean()
        if low > 0.1:
            w(f"  WARNING: {pct(low)} of windows matched with r < 0.95. The stored files may "
              "not come from these .vital files or this filter chain; treat Q1-Q7 with care.")

    # Q1
    w("\n[Q1] Horizon convention (positives). delta = pump onset - (window end + h min).")
    w("     delta ~ 0 s   -> window ENDS h min before infusion start (paper wording)")
    w("     delta ~ -30 s -> window BEGINS h min before infusion start")
    verdict_q1 = "UNCLEAR"
    if "onset_delta_free_sec" in pos.columns and pos.onset_delta_free_sec.notna().any():
        for h, g in pos.groupby("horizon"):
            d = g.onset_delta_free_sec.dropna()
            if len(d):
                w(f"     {h:>4.0f} min: median {d.median():+.1f} s  "
                  f"(n={len(d)}, within 2 s of 0: {pct((d.abs() <= TOL_SEC).mean())}, "
                  f"within 2 s of -30: {pct(((d + 30).abs() <= TOL_SEC).mean())})")
        d = pos.onset_delta_free_sec.dropna()
        f0 = (d.abs() <= TOL_SEC).mean()
        f30 = ((d + 30).abs() <= TOL_SEC).mean()
        if f0 >= 0.8:
            verdict_q1 = "ENDS h min before infusion start"
        elif f30 >= 0.8:
            verdict_q1 = "BEGINS h min before infusion start"
    w(f"  -> VERDICT Q1: {verdict_q1}")

    # Q2
    w("\n[Q2] Do the recovered filters reproduce stored ABP/ECG? (R^2 of a linear fit")
    w("     between the stored window and the v2 re-extraction; 1.000 = identical up to")
    w("     scaling. Block A = 'ends' convention, block B = 'begins' convention.)")
    rep_ok = []
    for m in ["abp", "ecg"]:
        for cls, g in (("positive", pos), ("control", ctl)):
            col = f"{m}_R2_A"
            if col in g.columns and g[col].notna().any():
                r = g[col].dropna()
                share = (r > 0.99).mean()
                rep_ok.append(share)
                w(f"     {m.upper()} {cls:8s}: median R^2 {r.median():.4f}, "
                  f">0.99 in {pct(share)} of {len(r)} windows")
    verdict_q2 = ("REPRODUCED" if rep_ok and min(rep_ok) >= 0.9 else
                  "PARTLY REPRODUCED" if rep_ok and max(rep_ok) >= 0.5 else "NOT REPRODUCED")
    w(f"  -> VERDICT Q2: {verdict_q2}")
    if {"abp_R2_A", "abp_R2_B"} <= set(win.columns):
        g = win[win.horizon == win.horizon.max()]
        if len(g):
            w(f"     At {g.horizon.max():.0f} min (block edge), median ABP R^2 "
              f"A={g.abp_R2_A.median():.4f} vs B={g.abp_R2_B.median():.4f}; "
              f"ECG A={g.get('ecg_R2_A', pd.Series(dtype=float)).median():.4f} vs "
              f"B={g.get('ecg_R2_B', pd.Series(dtype=float)).median():.4f}. "
              "The higher one supports that convention.")

    # Q3
    w("\n[Q3] PPG normalisation. For a per-window z-score: a*sd(window) = 1 and")
    w("     window offset = 0. For a z-score over the whole 15-min block:")
    w("     a*sd(block) = 1 and block offset = 0.")
    for cls, g in (("positive", pos), ("control", ctl)):
        if "ppg_a_x_sdwin" in g.columns and g.ppg_a_x_sdwin.notna().any():
            w(f"     PPG {cls:8s}: R^2 {g.ppg_R2_A.median():.4f} | a*sd(win) "
              f"{g.ppg_a_x_sdwin.median():.3f} | a*sd(block) {g.ppg_a_x_sdblock.median():.3f} | "
              f"offset win {g.ppg_offset_win.median():+.3f} | offset block "
              f"{g.ppg_offset_block.median():+.3f} | stored SD {g.ppg_stored_sd.median():.3f}")
            if cls == "positive":
                by_h = g.groupby("horizon").ppg_R2_A.median()
                w("       positive PPG R^2 by horizon: " +
                  ", ".join(f"{h:.0f} min {v:.3f}" for h, v in by_h.items()))
    w("  -> Interpret by hand (see instructions, section 7).")

    # Q4/Q5
    w("\n[Q4/Q5] BP targets: which 30-s segment of the numeric tracks matches the stored")
    w("        target exactly (max abs diff < 0.01 mmHg).")
    if not bp.empty and "target_max_abs_diff" in bp.columns:
        for cls, g in (("positive", bp[~bp.is_control]), ("control", bp[bp.is_control])):
            g = g[g.target_max_abs_diff.notna()]
            if g.empty:
                continue
            exact = (g.target_max_abs_diff < 0.01).mean()
            amb = (g.target_n_exact_matches > 1).mean()
            w(f"     {cls}: exact match found for {pct(exact)} of {len(g)} cases; "
              f"ambiguous (several exact matches) {pct(amb)}")
            col = ("target_start_minus_onset" if cls == "positive" else
                   "target_start_minus_block_end_B" if verdict_q1.startswith("BEGINS") else
                   "target_start_minus_block_end_A")
            if col in g.columns:
                vc = g[g.target_max_abs_diff < 0.01][col].round().value_counts().head(5)
                ref = "infusion start" if cls == "positive" else "end of the control block"
                w(f"       segment start relative to {ref} (s): " +
                  ", ".join(f"{int(k)} (n={v})" for k, v in vc.items()))
        w("     Positive expectation: -60 (segment from 60 s to 30 s before infusion start).")
        w("     Control: -30 = last 30 s of the control block (same rule as positives);")
        w("     -900 = first segment; anything else -> note the value.")
    else:
        w("     No BP target comparison was possible (see bp_target_check.csv).")

    # Q6
    w("\n[Q6] Dose rule. Share of positive windows whose implied onset is within 2 s of")
    if "onset_delta_rule_sec" in pos.columns:
        d_rule = pos.onset_delta_rule_sec.dropna()
        d_free = pos.onset_delta_free_sec.dropna()
        target = 0.0 if verdict_q1.startswith("ENDS") else (-30.0 if verdict_q1.startswith("BEGINS") else 0.0)
        w(f"     an onset from the recovered rule (>=1 mL/h, 60-s merge): "
          f"{pct(((d_rule - target).abs() <= TOL_SEC).mean()) if len(d_rule) else 'n/a'}")
        w(f"     any 0 -> >0 pump transition (no rule):                  "
          f"{pct(((d_free - target).abs() <= TOL_SEC).mean()) if len(d_free) else 'n/a'}")
    if allev is not None and not allev.empty and "rule_events" in allev.columns:
        ok = allev.dropna(subset=["rule_events"])
        w(f"     All positive cases: stored (case, dose) pairs {int(ok.stored_doses.sum())}, "
          f"recovered-rule events {int(ok.rule_events.sum())}; per-case counts equal in "
          f"{pct((ok.stored_doses == ok.rule_events).mean())} of {len(ok)} cases.")

    # Q7
    w("\n[Q7] Control placement.")
    c = bp[bp.is_control] if not bp.empty else bp
    if not c.empty and "block_end_frac_of_record" in c.columns:
        f = c.block_end_frac_of_record.dropna()
        w(f"     (block end + 30 s) / record length: median {f.median():.3f}, "
          f"IQR {f.quantile(.25):.3f}-{f.quantile(.75):.3f}, range {f.min():.3f}-{f.max():.3f}")
        if "block_mid_frac_of_op" in c.columns:
            g = c.block_mid_frac_of_op.dropna()
            if len(g):
                w(f"     block midpoint as fraction of surgery (opstart->opend): median "
                  f"{g.median():.3f}, IQR {g.quantile(.25):.3f}-{g.quantile(.75):.3f}")
        if "block_end_spread_across_horizons_sec" in c.columns:
            s = c.block_end_spread_across_horizons_sec.dropna()
            w(f"     Same block position at every horizon (spread <= 2 s): "
              f"{pct((s <= TOL_SEC).mean())} of {len(s)} control cases")
        w("     Narrow values near 0.5 -> surgical mid-point (thesis). Spread-out values")
        w("     matching 0.07-0.84 -> v3-style sampled placement.")

    L.append("\nPer-window details: windows_check.csv; BP targets: bp_target_check.csv;")
    L.append("events: events_check.csv; inventory: inventory.csv; errors: errors.csv")
    txt = "\n".join(L)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(txt)
    print(txt)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stored-dir", required=True,
                    help="Master_Training_Data_Expanded_New_2 (contains train/ val/ test/)")
    ap.add_argument("--bp-target-dir", required=True,
                    help="BP_target (contains sbp_target/ dbp_target/ map_target/)")
    ap.add_argument("--vital-dir", default=None,
                    help="folder with VitalDB .vital files; omit to download each case")
    ap.add_argument("--clinical-csv", default=None, help="VitalDB clinical_data.csv (optional)")
    ap.add_argument("--out", default="verify_results")
    ap.add_argument("--n-pos", type=int, default=20, help="positive (case, dose) pairs to check")
    ap.add_argument("--n-ctrl", type=int, default=20, help="control cases to check")
    ap.add_argument("--horizons", type=float, nargs="+", default=[3, 10, 15])
    ap.add_argument("--all-events", action="store_true",
                    help="also count infusion starts in ALL positive cases (Q6, slower)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    try:
        import vitaldb  # noqa: F401
    except ImportError:
        raise SystemExit("The 'vitaldb' package is missing: pip install vitaldb")

    print("Step 1/4  Reading the stored files ...")
    inv = build_inventory(args.stored_dir)
    inv.drop(columns=["path"]).to_csv(os.path.join(args.out, "inventory.csv"), index=False)

    clinical = None
    if args.clinical_csv:
        clinical = pd.read_csv(args.clinical_csv).set_index("caseid")

    rng = random.Random(args.seed)
    sel = inv[inv.horizon.isin(args.horizons)]
    pos_stems = sorted(sel[~sel.is_control].case_stem.unique())
    ctl_stems = sorted(sel[sel.is_control].case_stem.unique())
    pick = (rng.sample(pos_stems, min(args.n_pos, len(pos_stems))) +
            rng.sample(ctl_stems, min(args.n_ctrl, len(ctl_stems))))
    print(f"Step 2/4  Checking {len(pick)} cases at horizons {args.horizons} ...")

    win_rows, bp_rows, ev_rows, errors = [], [], [], []
    for stem in tqdm(pick, desc="cases"):
        g = sel[sel.case_stem == stem]
        try:
            w, b, e = check_case(stem, g, args, clinical)
            win_rows += w
            bp_rows.append(b)
            ev_rows.append(e)
        except Exception as ex:
            errors.append(dict(case_stem=stem, error=f"{type(ex).__name__}: {ex}",
                               trace=traceback.format_exc()[-500:]))
            print(f"  [!] {stem}: {type(ex).__name__}: {ex}")
        # save as we go, so a crash keeps partial results
        pd.DataFrame(win_rows).to_csv(os.path.join(args.out, "windows_check.csv"), index=False)
        pd.DataFrame(bp_rows).to_csv(os.path.join(args.out, "bp_target_check.csv"), index=False)
        pd.DataFrame(ev_rows).to_csv(os.path.join(args.out, "events_check.csv"), index=False)
        pd.DataFrame(errors).to_csv(os.path.join(args.out, "errors.csv"), index=False)

    allev = None
    if args.all_events:
        print("Step 3/4  Counting infusion starts in all positive cases ...")
        allev = count_all_events(inv, args)
        allev.to_csv(os.path.join(args.out, "all_events_count.csv"), index=False)
    else:
        print("Step 3/4  Skipped (add --all-events to count events in all positive cases)")

    print("Step 4/4  Writing summary ...\n")
    summarise(inv, pd.DataFrame(win_rows), pd.DataFrame(bp_rows),
              pd.DataFrame(ev_rows), allev, args.out)
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(dict(args=vars(args), n_cases=len(pick), n_errors=len(errors),
                       minutes=round((time.time() - t0) / 60, 1)), f, indent=2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Send the whole '{args.out}' folder back.")


if __name__ == "__main__":
    main()
