"""
Control extraction + distribution matching -- v3.

    meta_df, log_df  = run()                        # stage 1: extract + log
    matched, table1  = match_controls(ratio=5)      # stage 2: match (no waveform IO)

v3 changes (waveform processing now IDENTICAL to the latest VP script,
Vasopressor_Processed_Data_Raw_v2):
    1. Filters   : PPG 0.5-25 Hz band-pass, ECG 0.5-40 Hz band-pass,
                   ABP 25 Hz low-pass; 4th-order Butterworth, filtfilt (b,a).
    2. Gaps      : PCHIP for gaps <= 1 s, remaining NaN set to 0 (same as VP).
    3. Quality   : reject only if any signal is > 10% NaN (same as VP).
    4. Ephedrine : light-ephedrine ceiling lowered to 25.
    5. Logging   : every case gets a row in extraction_log.csv; gap handling
                   and quality-check rejects are printed as clean tables.
"""

import os
import re
import glob
import json
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

warnings.filterwarnings("ignore")

import vitaldb

try:
    from tqdm.notebook import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x


# ==========================================
# CONFIG
# ==========================================
TRIPLE_TRACK_JSON = r"D:\valid_triple_track_cases.json"
VP_LABELS_CSV     = r"D:\vasopressor_labels.csv"
CLINICAL_CSV      = r"D:\vitaldb\vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0\clinical_data.csv"
LOCAL_VITAL_DIR   = r"D:\vitaldb\vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0\vital_files"

OUTPUT_DIR        = r"F:\Control_Processed_Data_Matched_v3"   # NEW folder -- don't mix with older runs

SEED = 42

# ---- window / resampling (same as VP) ----
FS_RAW              = 500
FS_TARGET           = 125
WINDOW_MINS         = 15
TOTAL_SEC           = WINDOW_MINS * 60
SEGMENT_SEC         = 30
NUM_SEGMENTS        = TOTAL_SEC // SEGMENT_SEC
SAMPLES_PER_SEGMENT = SEGMENT_SEC * FS_TARGET
LEAD_OUT_SEC        = 30
MIN_CASE_DUR_SEC    = 1800

GAP_LIMIT_SEC         = 1.0
GAP_LIMIT_SAMPLES_RAW = int(GAP_LIMIT_SEC * FS_RAW)
MAX_NAN_FRAC          = 0.10          # same as VP sensor_present_check
FFILL_LIMIT_SEC       = 60

# ---- filters: IDENTICAL to the VP script (option B) ----
PPG_BAND    = (0.5, 25.0)
ECG_BAND    = (0.5, 40.0)
ABP_LOWPASS = 25.0

# ---- light-ephedrine window search ----
K_CANDIDATES           = 5
EPH_CEILING            = 25.0
USE_MAP_STABILITY_PICK = True   # True = among K candidates pick lowest MAP std (old behaviour)
                                # False = take the first valid candidate (random)

# ---- VP first-event onset fraction, empirical deciles (n=139) ----
VP_FRAC_DECILES = [0.069, 0.145, 0.222, 0.305, 0.351,
                   0.413, 0.488, 0.552, 0.621, 0.696, 0.839]

# ---- printing ----
MAX_PRINT_ROWS = None           # None = print every row of the gap / reject tables

# ---- tracks ----
TRACKS_WAVE = {"ppg": "SNUADC/PLETH", "ecg": "SNUADC/ECG_II", "abp": "SNUADC/ART"}
WAVE_ORDER  = ["ppg", "ecg", "abp"]      # same column order as VP

TRACKS_BASED_RAW = {
    "ppg": ["Solar8000/PLETH_SPO2", "Solar8000/PLETH_HR"],
    "ecg": ["Solar8000/HR"],
    "abp": ["Solar8000/ART_SBP", "Solar8000/ART_DBP", "Solar8000/ART_MBP",
            "EV1000/SV", "EV1000/SVV", "EV1000/CO",
            "Vigileo/SV", "Vigileo/SVV", "Vigileo/CO"],
}
TRACKS_BASED_FINAL = {
    "ppg": ["Solar8000/PLETH_SPO2", "Solar8000/PLETH_HR"],
    "ecg": ["Solar8000/HR"],
    "abp": ["Solar8000/ART_SBP", "Solar8000/ART_DBP", "Solar8000/ART_MBP",
            "Merged_SV", "Merged_SVV", "Merged_CO"],
}
MAP_TRACK = "Solar8000/ART_MBP"
FMS_FLOW  = "FMS/FLOW_RATE"

DISQUALIFYING_INFUSIONS = [
    "Orchestra/DOBU_RATE", "Orchestra/MRN_RATE",          # inotropes
    "Orchestra/NTG_RATE", "Orchestra/NPS_RATE",           # vasodilators
    "Orchestra/PGE1_RATE",
    "Orchestra/DEX2_RATE", "Orchestra/DEX4_RATE",         # dexmedetomidine
    "Orchestra/OXY_RATE", "Orchestra/DTZ_RATE", "Orchestra/AMD_RATE",
]
INFUSION_MIN_RATE = 0.01

OPTYPE_GROUPS = {
    "Transplantation": "HPB_transplant",
    "Hepatic": "HPB_transplant",
    "Biliary/Pancreas": "HPB_transplant",
    "Vascular": "Vascular",
    "Others": "Others",
}
DEFAULT_OPTYPE_GROUP = "GI_other"

COVARIATES = [
    "optype", "department", "approach", "position", "ane_type", "sex",
    "age", "bmi", "asa", "emop", "preop_htn", "preop_dm",
    "preop_hb", "preop_alb", "preop_cr", "preop_plt",
    "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca",
    "intraop_crystalloid", "intraop_colloid", "intraop_ebl",
    "intraop_rbc", "intraop_ffp", "intraop_uo", "intraop_ppf", "intraop_ftn",
    "icu_days", "death_inhosp",
]

NUMERIC_CLINICAL = [
    "casestart", "caseend", "opstart", "opend", "age", "bmi", "asa",
    "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca",
    "intraop_crystalloid", "intraop_colloid", "intraop_ebl", "intraop_rbc",
    "intraop_ffp", "intraop_uo", "intraop_ppf", "intraop_ftn",
    "preop_hb", "preop_alb", "preop_cr", "preop_plt", "icu_days",
]

# Reject reasons that come from the waveform quality check
QC_PREFIX = "qc_"


# ==========================================
# SIGNAL HELPERS -- copied from the VP script
# ==========================================
def apply_bandpass_filter(data, lowcut, highcut, fs=FS_RAW, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band", analog=False)
    return filtfilt(b, a, data)


def apply_lowpass_filter(data, cutoff=ABP_LOWPASS, fs=FS_RAW, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data)


def fill_gaps_and_zeropad(signal_array):
    """Same output as the VP version. Also returns how many samples were
    interpolated and how many were set to zero, for the log."""
    n_nan = int(np.isnan(signal_array).sum())
    s = pd.Series(signal_array)
    s = s.interpolate(method="pchip", limit=GAP_LIMIT_SAMPLES_RAW,
                      limit_direction="forward")
    n_zero = int(s.isna().sum())
    stats = {"n_nan": n_nan, "n_interp": n_nan - n_zero, "n_zero": n_zero}
    return s.fillna(0.0).values, stats


def sensor_present_check(ppg, ecg, abp, max_nan_frac=MAX_NAN_FRAC):
    """Same rule as VP (reject if any signal > 10% NaN). Also returns which
    signal failed, in the same order VP checks them (abp, ecg, ppg)."""
    for name, sig in (("abp", abp), ("ecg", ecg), ("ppg", ppg)):
        if np.isnan(sig).mean() > max_nan_frac:
            return False, f"{QC_PREFIX}nan_over_{int(max_nan_frac*100)}pct_{name}"
    return True, "ok"


def merge_ev1000_vigileo(df):
    """Merge BEFORE any forward-fill, so Vigileo can contribute."""
    for param in ["SV", "SVV", "CO"]:
        ev, vg, mg = f"EV1000/{param}", f"Vigileo/{param}", f"Merged_{param}"
        if ev in df.columns and vg in df.columns:
            df[mg] = df[ev].combine_first(df[vg])
        elif ev in df.columns:
            df[mg] = df[ev]
        elif vg in df.columns:
            df[mg] = df[vg]
        else:
            df[mg] = np.nan
    return df


def extract_15_points(data_segment, target_len=15):
    clean = data_segment[~np.isnan(data_segment)]
    if len(clean) < 2:
        return [np.nan] * target_len
    idx = np.linspace(0, len(clean) - 1, target_len).astype(int)
    return clean[idx].tolist()


# ==========================================
# POOL
# ==========================================
def load_clinical(path=CLINICAL_CSV):
    cd = pd.read_csv(path)
    for c in NUMERIC_CLINICAL:
        if c in cd.columns:
            cd[c] = pd.to_numeric(cd[c], errors="coerce")
    cd["case_dur"] = cd["caseend"] - cd["casestart"]
    cd["optype_group"] = cd["optype"].map(
        lambda x: OPTYPE_GROUPS.get(x, DEFAULT_OPTYPE_GROUP))
    return cd


def build_file_index(vital_dir=LOCAL_VITAL_DIR):
    index = {}
    for f in glob.glob(os.path.join(vital_dir, "**", "*.vital"), recursive=True):
        m = re.search(r"(\d+)", os.path.basename(f).replace(".vital", ""))
        if m:
            index[int(m.group(1))] = f
    return index


def build_pool():
    cd = load_clinical()
    with open(TRIPLE_TRACK_JSON) as f:
        triple = set(int(c) for c in json.load(f))
    vp_cases = set(pd.read_csv(VP_LABELS_CSV)["case_id"].astype(int))

    eph = cd["intraop_eph"].fillna(0)
    phe = cd["intraop_phe"].fillna(0)
    epi = cd["intraop_epi"].fillna(0)
    ca = cd["intraop_ca"].fillna(0) if "intraop_ca" in cd else pd.Series(0, index=cd.index)

    steps = []
    m = cd;                                   steps.append(("all VitalDB cases", len(m)))
    m = m[m["caseid"].isin(triple)];          steps.append(("has PPG + ECG + ABP", len(m)))
    m = m[~m["caseid"].isin(vp_cases)];       steps.append(("not a vasopressor case", len(m)))
    m = m[phe.loc[m.index] == 0];             steps.append(("phenylephrine = 0", len(m)))
    m = m[epi.loc[m.index] == 0];             steps.append(("epinephrine = 0", len(m)))
    m = m[eph.loc[m.index] <= EPH_CEILING];   steps.append((f"ephedrine <= {EPH_CEILING:g} mg", len(m)))
    m = m[ca.loc[m.index] == 0];              steps.append(("calcium = 0", len(m)))
    m = m[m["ane_type"] == "General"];        steps.append(("general anaesthesia", len(m)))
    m = m[m["aline1"].notna()];               steps.append(("arterial line", len(m)))
    m = m[m["case_dur"] >= MIN_CASE_DUR_SEC]; steps.append((f"case >= {MIN_CASE_DUR_SEC // 60} min", len(m)))

    print_header("STEP 1  Case selection (clinical data)")
    prev = None
    for label, n in steps:
        drop = "" if prev is None else f"(-{prev - n})"
        print(f"  {label:<28}{n:>7}   {drop}")
        prev = n
    return m.reset_index(drop=True)


def has_disqualifying_infusion(vf, names):
    present = [t for t in DISQUALIFYING_INFUSIONS if t in names]
    if not present:
        return False, None
    df = vf.to_pandas(present, interval=1)
    for t in present:
        if t in df.columns and np.nansum(df[t].values > INFUSION_MIN_RATE) > 0:
            return True, t.split("/")[-1].replace("_RATE", "")
    return False, None


def fms_active_mask(vf, names):
    if FMS_FLOW not in names:
        return None
    df = vf.to_pandas([FMS_FLOW], interval=1)
    if FMS_FLOW not in df.columns:
        return None
    v = df[FMS_FLOW].values
    return np.nan_to_num(v, nan=0.0) > 0 if v.size else None


# ==========================================
# WINDOW PLACEMENT
# ==========================================
def sample_fraction(rng):
    u = rng.random() * (len(VP_FRAC_DECILES) - 1)
    lo = int(np.floor(u))
    hi = min(lo + 1, len(VP_FRAC_DECILES) - 1)
    w = u - lo
    return VP_FRAC_DECILES[lo] * (1 - w) + VP_FRAC_DECILES[hi] * w


def frac_to_window(frac, case_dur):
    end = frac * case_dur - LEAD_OUT_SEC
    start = end - TOTAL_SEC
    return (start, end) if (start >= 0 and end <= case_dur) else None


def candidate_windows(case_dur, n, rng):
    out = []
    for _ in range(n * 4):
        w = frac_to_window(sample_fraction(rng), case_dur)
        if w and all(abs(w[0] - o[0]) > SEGMENT_SEC for o in out):
            out.append(w)
        if len(out) >= n:
            break
    if not out:
        s = case_dur / 2 - TOTAL_SEC / 2
        if s >= 0 and s + TOTAL_SEC <= case_dur:
            out.append((s, s + TOTAL_SEC))
    return out


def window_overlaps_fms(win, mask):
    if mask is None:
        return False
    s, e = int(win[0]), int(min(win[1], len(mask)))
    return s < len(mask) and bool(mask[s:e].any())


def map_std_for_window(map_series, win):
    if map_series is None:
        return np.nan
    s, e = int(win[0]), int(min(win[1], len(map_series)))
    seg = map_series[s:e]
    seg = seg[~np.isnan(seg)]
    return float(np.std(seg)) if seg.size >= 30 else np.nan


# ==========================================
# PER-CASE
# ==========================================
def process_case(case_id, path, row, rng, out_dir, dry_run=False):
    """Returns (status, reason, meta, qc). qc holds NaN / gap stats per signal."""
    case_dur = float(row["case_dur"])
    eph = float(row.get("intraop_eph", 0) or 0)
    qc = {}

    vf = vitaldb.VitalFile(path)
    names = set(vf.get_track_names())

    for k, t in TRACKS_WAVE.items():
        if t not in names:
            return "reject", f"missing_{k}_track", None, qc

    bad, drug = has_disqualifying_infusion(vf, names)
    if bad:
        return "reject", f"infusion_{drug}", None, qc

    mask = fms_active_mask(vf, names)

    n_cand = K_CANDIDATES if 0 < eph <= EPH_CEILING else 1
    cands = [w for w in candidate_windows(case_dur, n_cand, rng)
             if not window_overlaps_fms(w, mask)]
    if not cands:
        return "reject", "no_valid_window", None, qc

    map_series = None
    if USE_MAP_STABILITY_PICK and len(cands) > 1 and MAP_TRACK in names:
        d = vf.to_pandas([MAP_TRACK], interval=1)
        if MAP_TRACK in d.columns:
            map_series = d[MAP_TRACK].values

    if USE_MAP_STABILITY_PICK and len(cands) > 1:
        stds = [map_std_for_window(map_series, w) for w in cands]
        if np.all(np.isnan(stds)):
            win, chosen_std = cands[0], np.nan
        else:
            i = int(np.nanargmin(stds))
            win, chosen_std = cands[i], stds[i]
    else:
        win, chosen_std = cands[0], np.nan

    start_sec, end_sec = win
    meta = {
        "case_id": case_id,
        "case_dur_sec": case_dur,
        "window_start_sec": round(start_sec, 2),
        "window_end_sec": round(end_sec, 2),
        "window_frac": round((end_sec + LEAD_OUT_SEC) / case_dur, 4),
        "n_candidates": len(cands),
        "map_std": chosen_std,
        "optype_group": row.get("optype_group"),
    }
    for c in COVARIATES:
        if c in row.index:
            meta[c] = row[c]

    if dry_run:
        return "ok", "dry_run", meta, qc

    # ---- load the window (same as VP) ----
    idx_s, idx_e = int(start_sec * FS_RAW), int(end_sec * FS_RAW)
    arr = vf.to_numpy([TRACKS_WAVE[m] for m in WAVE_ORDER], interval=1 / FS_RAW)
    if arr.shape[0] < idx_e:
        return "reject", "window_past_array_end", None, qc
    ppg_raw = arr[idx_s:idx_e, 0]
    ecg_raw = arr[idx_s:idx_e, 1]
    abp_raw = arr[idx_s:idx_e, 2]

    for nm, sig in (("ppg", ppg_raw), ("ecg", ecg_raw), ("abp", abp_raw)):
        qc[f"{nm}_nan_pct"] = round(100 * float(np.isnan(sig).mean()), 3)

    # ---- quality check (same as VP) ----
    ok, why = sensor_present_check(ppg_raw, ecg_raw, abp_raw)
    if not ok:
        return "reject", why, None, qc

    # ---- preprocess (same order and settings as VP) ----
    ppg_filled, g_ppg = fill_gaps_and_zeropad(ppg_raw)
    ecg_filled, g_ecg = fill_gaps_and_zeropad(np.clip(ecg_raw, -5.0, 5.0))
    abp_filled, g_abp = fill_gaps_and_zeropad(np.clip(abp_raw, 0.0, 350.0))
    for nm, g in (("ppg", g_ppg), ("ecg", g_ecg), ("abp", g_abp)):
        qc[f"{nm}_interp_s"] = round(g["n_interp"] / FS_RAW, 3)
        qc[f"{nm}_zero_s"]   = round(g["n_zero"] / FS_RAW, 3)

    waves = {
        "ppg": resample_poly(apply_bandpass_filter(ppg_filled, *PPG_BAND), FS_TARGET, FS_RAW),
        "ecg": resample_poly(apply_bandpass_filter(ecg_filled, *ECG_BAND), FS_TARGET, FS_RAW),
        "abp": resample_poly(apply_lowpass_filter(abp_filled, ABP_LOWPASS), FS_TARGET, FS_RAW),
    }

    # ---- based tracks: merge first, then limited fill ----
    avail = [t for t in sum(TRACKS_BASED_RAW.values(), []) if t in names]
    df_based = pd.DataFrame()
    if avail:
        df_based = vf.to_pandas(avail, interval=1)
        df_based = merge_ev1000_vigileo(df_based)
        df_based = df_based.ffill(limit=FFILL_LIMIT_SEC).bfill(limit=FFILL_LIMIT_SEC)

    datasets = {f"{m}_{d}": [] for m in ["ppg", "ecg", "abp"] for d in ["wave", "based"]}
    for si in range(NUM_SEGMENTS):
        t0, t1 = si * SEGMENT_SEC, (si + 1) * SEGMENT_SEC
        abs0, abs1 = start_sec + t0, start_sec + t1
        base = [case_id, 0, 0, "Control", round(abs0, 2), f"{t0 // 60}m{t0 % 60:02d}s"]

        for mod in ["ppg", "ecg", "abp"]:
            s, e = int(t0 * FS_TARGET), int(t1 * FS_TARGET)
            w = waves[mod][s:e]
            if len(w) != SAMPLES_PER_SEGMENT:
                w = np.pad(w, (0, max(0, SAMPLES_PER_SEGMENT - len(w))),
                           "edge")[:SAMPLES_PER_SEGMENT]
            datasets[f"{mod}_wave"].append(base + w.tolist())

        for mod in ["ppg", "ecg", "abp"]:
            vals = []
            for t in TRACKS_BASED_FINAL[mod]:
                if t in df_based.columns:
                    sl = df_based[(df_based.index >= abs0) &
                                  (df_based.index < abs1)][t].values
                    vals.extend(extract_15_points(sl))
                else:
                    vals.extend([np.nan] * 15)
            datasets[f"{mod}_based"].append(base + vals)

    base_cols = ["case_id", "label", "label_binary", "vasopressor_name",
                 "segment_start_sec", "time_offset"]
    for mod in ["ppg", "ecg", "abp"]:
        wcols = base_cols + [f"{mod}_{i}" for i in range(SAMPLES_PER_SEGMENT)]
        bcols = base_cols.copy()
        for t in TRACKS_BASED_FINAL[mod]:
            nm = t.split("/")[-1].lower()
            bcols += [f"{nm}_{i}" for i in range(15)]
        pd.DataFrame(datasets[f"{mod}_wave"], columns=wcols).to_csv(
            os.path.join(out_dir, mod, f"Case_{case_id:04d}_control_{mod}_wave.csv"),
            index=False)
        pd.DataFrame(datasets[f"{mod}_based"], columns=bcols).to_csv(
            os.path.join(out_dir, mod, f"Case_{case_id:04d}_control_{mod}_based.csv"),
            index=False)

    return "ok", "extracted", meta, qc


# ==========================================
# PRINT HELPERS
# ==========================================
def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_table(df, max_rows=MAX_PRINT_ROWS):
    if df.empty:
        print("  (none)")
        return
    shown = df if max_rows is None else df.head(max_rows)
    print(shown.to_string(index=False))
    if max_rows is not None and len(df) > max_rows:
        print(f"  ... {len(df) - max_rows} more rows in extraction_log.csv")


def print_reports(log_df, meta_df, n_attempted):
    ok   = log_df[log_df.status == "ok"]
    rej  = log_df[log_df.status == "reject"]
    err  = log_df[log_df.status == "error"]
    qc   = rej[rej.reason.str.startswith(QC_PREFIX)]
    case = rej[~rej.reason.str.startswith(QC_PREFIX)]

    # ---------- overall ----------
    print_header("STEP 2  Extraction summary")
    print(f"  Cases attempted                {n_attempted:>6}")
    print(f"  Extracted                      {len(ok):>6}")
    print(f"  Rejected (case-level)          {len(case):>6}")
    print(f"  Rejected (quality check)       {len(qc):>6}")
    print(f"  Errors                         {len(err):>6}")

    print("\n  Case-level reject reasons")
    for k, v in case.reason.value_counts().items():
        print(f"    {k:<32}{v:>6}")
    if case.empty:
        print("    (none)")

    # ---------- quality check ----------
    print_header(f"STEP 3  Quality-check rejects  (any signal > {int(MAX_NAN_FRAC*100)}% missing)")
    print(f"  Rule: same as VP script. {len(qc)} windows rejected.\n")
    if not qc.empty:
        for k, v in qc.reason.value_counts().items():
            print(f"    {k:<32}{v:>6}")
        print()
        t = qc[["case_id", "reason", "ppg_nan_pct", "ecg_nan_pct", "abp_nan_pct"]].copy()
        t.insert(1, "file", t.case_id.map(lambda c: f"Case_{c:04d}_control"))
        t.columns = ["case_id", "file", "failed_on", "PPG_%NaN", "ECG_%NaN", "ABP_%NaN"]
        t["failed_on"] = t["failed_on"].str.split("_").str[-1].str.upper()
        print_table(t.sort_values("case_id"))
    else:
        print("  (none)")

    # ---------- gap handling ----------
    print_header("STEP 4  Gap handling in extracted windows")
    print(f"  Gaps <= {GAP_LIMIT_SEC:g} s: PCHIP interpolation.  Longer gaps: set to 0 (same as VP).")
    print("  Values are seconds of signal inside the 15-min window.\n")

    gcols = [f"{m}_{k}" for m in ["ppg", "ecg", "abp"] for k in ["interp_s", "zero_s"]]
    if ok.empty:
        print("  (no extracted windows)")
        return
    g = ok[["case_id"] + gcols].fillna(0)

    print(f"  {'signal':<8}{'windows with any gap':>22}{'with zero-filled part':>24}")
    for m in ["ppg", "ecg", "abp"]:
        n_any  = int(((g[f"{m}_interp_s"] + g[f"{m}_zero_s"]) > 0).sum())
        n_zero = int((g[f"{m}_zero_s"] > 0).sum())
        print(f"  {m.upper():<8}{n_any:>14} / {len(g):<6}{n_zero:>16} / {len(g):<6}")

    gapped = g[(g[gcols] > 0).any(axis=1)].copy()
    print(f"\n  Windows that needed gap handling: {len(gapped)} / {len(g)}\n")
    if not gapped.empty:
        gapped.insert(1, "file", gapped.case_id.map(lambda c: f"Case_{c:04d}_control"))
        gapped.columns = ["case_id", "file",
                          "PPG_interp_s", "PPG_zero_s",
                          "ECG_interp_s", "ECG_zero_s",
                          "ABP_interp_s", "ABP_zero_s"]
        print_table(gapped.sort_values("case_id"))

    # ---------- window placement ----------
    if not meta_df.empty:
        print_header("STEP 5  Window placement check")
        print(f"  Window fraction : median {meta_df.window_frac.median():.3f}   (VP 0.413)")
        print(f"  Case duration   : median {meta_df.case_dur_sec.median():.0f} s   (VP 19109)")
        print(f"  K={K_CANDIDATES} search      : {int((meta_df.n_candidates > 1).sum())} light-ephedrine cases"
              f"  (MAP-stability pick {'ON' if USE_MAP_STABILITY_PICK else 'OFF'})")


# ==========================================
# STAGE 1
# ==========================================
def run(limit=0, dry_run=False, seed=SEED, out_dir=OUTPUT_DIR):
    for sub in ["ppg", "ecg", "abp"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    pool = build_pool()
    file_index = build_file_index()
    pool = pool[pool["caseid"].isin(file_index.keys())].copy()
    print(f"  {'.vital file on disk':<28}{len(pool):>7}")

    ids = pool["caseid"].tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)
    if limit:
        ids = ids[:limit]

    pool = pool.set_index("caseid")
    metas, logs = [], []

    for cid in tqdm(ids, desc="controls"):
        try:
            status, reason, meta, qc = process_case(
                cid, file_index[cid], pool.loc[cid], rng, out_dir, dry_run)
        except Exception as e:
            status, reason, meta, qc = "error", f"{type(e).__name__}: {str(e)[:60]}", None, {}
        if status == "ok" and meta:
            metas.append(meta)
        logs.append({"case_id": cid, "status": status, "reason": reason, **qc})

    meta_df = pd.DataFrame(metas)
    log_df = pd.DataFrame(logs)
    for c in ["ppg_nan_pct", "ecg_nan_pct", "abp_nan_pct"] + \
             [f"{m}_{k}" for m in ["ppg", "ecg", "abp"] for k in ["interp_s", "zero_s"]]:
        if c not in log_df.columns:
            log_df[c] = np.nan

    if not meta_df.empty:
        meta_df.to_csv(os.path.join(out_dir, "control_manifest.csv"), index=False)
    log_df.to_csv(os.path.join(out_dir, "extraction_log.csv"), index=False)

    print_reports(log_df, meta_df, len(ids))
    print(f"\n  Saved: control_manifest.csv, extraction_log.csv  ->  {out_dir}")
    return meta_df, log_df


# ==========================================
# STAGE 2 -- matching, no waveform IO
# ==========================================
def _vp_frame():
    cd = load_clinical()
    vp = pd.read_csv(VP_LABELS_CSV)
    vp["case_id"] = vp["case_id"].astype(int)
    m = vp.merge(cd, left_on="case_id", right_on="caseid")
    return m.sort_values(["case_id", "start_sec"]).groupby("case_id").first().reset_index()


def match_controls(ratio=5, caliper=0.25, fallback_caliper=0.50,
                   seed=SEED, out_dir=OUTPUT_DIR):
    """Match controls to VP cases on optype group + case_dur caliper.
    Falls back to a wider caliper, then duration-only. No control is reused."""
    man = pd.read_csv(os.path.join(out_dir, "control_manifest.csv"))
    vp = _vp_frame()
    rng = random.Random(seed)

    used, rows = set(), []
    for _, v in vp.iterrows():
        lo, hi = (1 - caliper) * v.case_dur, (1 + caliper) * v.case_dur
        lo2, hi2 = (1 - fallback_caliper) * v.case_dur, (1 + fallback_caliper) * v.case_dur
        cand = man[(~man.case_id.isin(used)) &
                   (man.optype_group == v.optype_group) &
                   (man.case_dur_sec.between(lo, hi))]
        tier = "optype + dur +/-25%"
        if len(cand) < ratio:
            cand = man[(~man.case_id.isin(used)) &
                       (man.optype_group == v.optype_group) &
                       (man.case_dur_sec.between(lo2, hi2))]
            tier = "optype + dur +/-50%"
        if len(cand) < ratio:
            cand = man[(~man.case_id.isin(used)) &
                       (man.case_dur_sec.between(lo2, hi2))]
            tier = "dur +/-50% only"
        if cand.empty:
            continue
        take = rng.sample(list(cand.case_id), min(ratio, len(cand)))
        used.update(take)
        for cid in take:
            rows.append({"vp_case_id": v.case_id, "control_case_id": cid,
                         "match_tier": tier, "vp_case_dur": v.case_dur,
                         "vp_optype_group": v.optype_group})

    matched = pd.DataFrame(rows)
    matched.to_csv(os.path.join(out_dir, "matched_controls.csv"), index=False)
    sel = man[man.case_id.isin(matched.control_case_id)]

    print_header(f"STEP 6  Matching  (target {ratio}:1)")
    print(f"  Matched {len(matched)} controls to {matched.vp_case_id.nunique()} / {len(vp)} VP cases\n")
    print("  Match tier")
    for k, v in matched.match_tier.value_counts().items():
        print(f"    {k:<26}{v:>6}")

    print_header("Table 1  (median [IQR] or proportion)")
    print(f"  {'variable':<22}{'VP':>22}{'matched controls':>24}")
    for c in ["case_dur", "age", "bmi", "preop_hb", "preop_alb",
              "intraop_crystalloid", "intraop_ebl", "intraop_uo"]:
        a = vp[c].dropna()
        b = sel[c if c != "case_dur" else "case_dur_sec"].dropna()
        if len(a) and len(b):
            fa = f"{a.median():.0f} [{a.quantile(.25):.0f}-{a.quantile(.75):.0f}]"
            fb = f"{b.median():.0f} [{b.quantile(.25):.0f}-{b.quantile(.75):.0f}]"
            print(f"  {c:<22}{fa:>22}{fb:>24}")
    for c in ["emop", "preop_htn", "preop_dm"]:
        if c in vp and c in sel:
            print(f"  {c:<22}{vp[c].mean():>22.3f}{sel[c].mean():>24.3f}")

    table1 = pd.DataFrame({"vp": vp.describe().T.get("50%"),
                           "control": sel.describe().T.get("50%")})
    return matched, table1


if __name__ == "__main__":
    meta_df, log_df = run()
    matched, table1 = match_controls(ratio=5)