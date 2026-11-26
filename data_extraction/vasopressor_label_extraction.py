"""
Vasopressor infusion label extraction from VitalDB .vital files.

Self-contained: every timestamp is derived from the .vital file itself.
No clinical_data.csv dependency.

Time frame: all *_sec values are seconds from the .vital record origin
(index 0 of a 1 Hz to_numpy call), matching the casestart = 0 convention
used by clinical_data.csv, so results can be joined on case_id later.

Original six output columns are preserved unchanged; everything else is
additive.

NOT derivable from the .vital file, and therefore absent here:
  opstart / opend  -> no surgical-phase (pre-incision vs intraop) flag
  weight           -> no mg/kg normalisation
  asa, optype, ebl -> join from clinical_data.csv downstream if needed
"""

import os
import glob
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import vitaldb

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DIR = r"D:\vitaldb\vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0\vital_files"
OUTPUT_CSV = r"D:/vasopressor_labels.csv"

TARGET_DRUGS = ['PHEN_RATE', 'NEPI_RATE', 'EPI_RATE', 'VASO_RATE', 'DOPA_RATE']

# Clinical minimum infusion rate, mL/hr, per drug.
# At 20 mcg/mL, 1 mL/hr delivers ~0.005 mcg/kg/min norepinephrine -- about
# a tenth of a starting dose. Below this is keep-vein-open or residual
# pump reading, not treatment.
MIN_RATE_BY_DRUG = {
    'NEPI_RATE': 1.0,   # norepinephrine 20 mcg/mL
    'EPI_RATE':  1.0,   # epinephrine 20 mcg/mL
    'PHEN_RATE': 1.0,   # phenylephrine 100 mcg/mL
    'DOPA_RATE': 1.0,   # dopamine 2 mg/mL
    'VASO_RATE': 0.5,   # vasopressin 0.2 U/mL
}
DEFAULT_MIN_RATE = 1.0

# No induction / emergence exclusion is applied. Every detected event is
# written out. Cut by hand from the CSV using anestart_to_pressor_sec and
# pressor_to_aneend_sec, which are raw margins with no threshold baked in.
MIN_EVENT_SEC = 10
GAP_MERGE_SEC = 60

DEPTH_OFFSETS = [-900, -600, -300, -60, 0]   # sec relative to infusion onset
DEPTH_HALFWIN = 30           # median over +/- 30 s at each offset
PREWIN_SEC = 900             # 15-min pre-window for dose and coverage

# Sustained-activity rule for deriving the anesthesia window: a candidate
# index qualifies only if at least SUSTAIN_FRAC of the following
# SUSTAIN_WIN seconds are also active. Stops a single spike from
# anchoring the whole case.
SUSTAIN_WIN = 60
SUSTAIN_FRAC = 0.8

# ==========================================
# TRACKS
# ==========================================
DEPTH_TRACKS = {
    'mac':       'Primus/MAC',
    'bis':       'BIS/BIS',
    'bis_sqi':   'BIS/SQI',
    'ppf_ce':    'Orchestra/PPF20_CE',
    'rftn20_ce': 'Orchestra/RFTN20_CE',
    'rftn50_ce': 'Orchestra/RFTN50_CE',
}

AGENT_TRACKS = {
    'exp_sevo': 'Primus/EXP_SEVO',      # kPa
    'exp_des':  'Primus/EXP_DES',       # kPa
    'fin2o':    'Primus/FIN2O',         # %
    'gas2_exp': 'Solar8000/GAS2_EXPIRED',
}

DOSE_TRACKS = {
    'ppf_rate':   'Orchestra/PPF20_RATE',
    'ppf_vol':    'Orchestra/PPF20_VOL',     # mL cumulative -> x20 = mg
    'rftn20_vol': 'Orchestra/RFTN20_VOL',    # mL cumulative -> x20 = mcg
    'rftn50_vol': 'Orchestra/RFTN50_VOL',    # mL cumulative -> x50 = mcg
    'dex2_rate':  'Orchestra/DEX2_RATE',
    'dex4_rate':  'Orchestra/DEX4_RATE',
}

AIRWAY_TRACKS = {
    'etco2_primus': 'Primus/ETCO2',
    'etco2_solar':  'Solar8000/ETCO2',
}

# Threshold above which a track counts as "anesthetic being delivered"
AGENT_ACTIVE_THRESH = {
    'mac':      0.10,   # unitless MAC
    'exp_sevo': 0.10,   # kPa
    'exp_des':  0.10,   # kPa
    'ppf_ce':   0.10,   # mcg/mL effect-site
    'ppf_rate': 0.10,   # mL/hr
}
ETCO2_ACTIVE_THRESH = 5.0    # mmHg -- airway connected and ventilating
BIS_SQI_MIN = 50             # BIS below this signal quality is unreliable

EMPTY = np.array([np.nan])


# ==========================================
# HELPERS
# ==========================================
def window_median(arr, centre, halfwin):
    """Median over [centre-halfwin, centre+halfwin], NaN-safe."""
    lo = max(0, int(centre) - halfwin)
    hi = min(len(arr), int(centre) + halfwin + 1)
    if lo >= hi:
        return np.nan
    seg = arr[lo:hi]
    if np.all(np.isnan(seg)):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmedian(seg))


def coverage(arr, lo, hi):
    """Fraction of non-NaN samples in [lo, hi)."""
    lo, hi = max(0, int(lo)), min(len(arr), int(hi))
    if lo >= hi:
        return 0.0
    return float(np.mean(~np.isnan(arr[lo:hi])))


def cumulative_delta(arr, lo, hi):
    """Increase in a cumulative-volume track across [lo, hi]."""
    lo, hi = max(0, int(lo)), min(len(arr) - 1, int(hi))
    if lo >= hi:
        return np.nan
    seg = arr[lo:hi + 1]
    valid = seg[~np.isnan(seg)]
    if valid.size < 2:
        return np.nan
    return float(max(0.0, valid[-1] - valid[0]))


def sustained_bounds(mask, win=SUSTAIN_WIN, frac=SUSTAIN_FRAC):
    """
    First and last index of sustained activity in a boolean mask.

    An index qualifies as a start if >= frac of the next `win` samples are
    active, and as an end if >= frac of the preceding `win` samples are.
    Returns (None, None) when nothing qualifies.
    """
    m = np.asarray(mask, dtype=float)
    if m.size == 0 or not m.any():
        return None, None
    if m.size < win:
        idx = np.where(m > 0)[0]
        return int(idx[0]), int(idx[-1])

    # rolling forward mean via cumulative sum
    cs = np.concatenate([[0.0], np.cumsum(m)])
    fwd = (cs[win:] - cs[:-win]) / win          # mean of [i, i+win)
    ok = np.where(fwd >= frac)[0]
    if ok.size == 0:
        idx = np.where(m > 0)[0]
        return int(idx[0]), int(idx[-1])
    start = int(ok[0])
    end = int(ok[-1] + win - 1)
    return start, min(end, m.size - 1)


def derive_anesthesia_window(sig, n_rec):
    """
    Anesthesia start/end in record seconds, derived from drug delivery and
    airway signals only.

    Returns (start, end, start_src, end_src, n_sources).
    """
    candidates = {}

    for key, thresh in AGENT_ACTIVE_THRESH.items():
        arr = sig.get(key)
        if arr is None or arr.size < 2:
            continue
        mask = np.nan_to_num(arr, nan=-1.0) > thresh
        s, e = sustained_bounds(mask)
        if s is not None:
            candidates[key] = (s, e)

    for key in AIRWAY_TRACKS:
        arr = sig.get(key)
        if arr is None or arr.size < 2:
            continue
        mask = np.nan_to_num(arr, nan=-1.0) > ETCO2_ACTIVE_THRESH
        s, e = sustained_bounds(mask)
        if s is not None:
            candidates[key] = (s, e)

    if not candidates:
        return 0, n_rec - 1, 'record_fallback', 'record_fallback', 0

    start_src = min(candidates, key=lambda k: candidates[k][0])
    end_src = max(candidates, key=lambda k: candidates[k][1])
    return (candidates[start_src][0], candidates[end_src][1],
            start_src, end_src, len(candidates))


def detect_events(values, min_rate, gap_sec, min_dur):
    """Runs above min_rate, merged across gaps <= gap_sec."""
    active = np.where((values > min_rate) & (~np.isnan(values)))[0]
    if active.size == 0:
        return []
    breaks = np.where(np.diff(active) > gap_sec)[0]
    starts = np.insert(active[breaks + 1], 0, active[0])
    ends = np.append(active[breaks], active[-1])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_dur]


def classify_anesthetic(sig, onset):
    """Inhalational / TIVA / mixed, from track presence in the pre-window."""
    lo = max(0, onset - PREWIN_SEC)
    inhal = max(coverage(sig.get('mac', EMPTY), lo, onset),
                coverage(sig.get('exp_sevo', EMPTY), lo, onset),
                coverage(sig.get('exp_des', EMPTY), lo, onset))
    tiva = max(coverage(sig.get('ppf_ce', EMPTY), lo, onset),
               coverage(sig.get('ppf_rate', EMPTY), lo, onset))
    if inhal > 0.5 and tiva > 0.5:
        return 'mixed'
    if inhal > 0.5:
        return 'inhalational'
    if tiva > 0.5:
        return 'tiva'
    return 'unknown'


def volatile_agent(sig, onset):
    lo = max(0, onset - PREWIN_SEC)
    sevo = coverage(sig.get('exp_sevo', EMPTY), lo, onset)
    des = coverage(sig.get('exp_des', EMPTY), lo, onset)
    if sevo > 0.5 and des > 0.5:
        return 'both'
    if sevo > 0.5:
        return 'sevoflurane'
    if des > 0.5:
        return 'desflurane'
    return 'none'


# ==========================================
# MAIN
# ==========================================
def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**/*.vital"), recursive=True))
    if not files:
        print(f"No .vital files found under {RAW_DIR}")
        return
    print(f"Found {len(files)} files")

    all_events = []
    skipped = Counter()
    failed = []
    t0 = time.time()

    aux_map = {**DEPTH_TRACKS, **AGENT_TRACKS, **DOSE_TRACKS, **AIRWAY_TRACKS}

    for i, file_path in enumerate(files):
        stem = os.path.basename(file_path).replace('.vital', '')
        try:
            case_id = int(stem)
        except ValueError:
            skipped['bad_filename'] += 1
            continue

        try:
            vf = vitaldb.VitalFile(file_path)
            available = set(vf.get_track_names())
        except Exception as exc:
            failed.append((case_id, 'open_failed', repr(exc)))
            continue

        drug_tracks = sorted(t for t in available
                             if any(k in t.upper() for k in TARGET_DRUGS))
        if not drug_tracks:
            skipped['no_pressor_track'] += 1
            continue

        aux_present = {k: v for k, v in aux_map.items() if v in available}
        request = drug_tracks + list(aux_present.values())

        try:
            data = vf.to_numpy(request, interval=1)   # 1 Hz -> index == seconds
        except Exception as exc:
            failed.append((case_id, 'to_numpy_failed', repr(exc)))
            continue

        if data is None or data.ndim != 2 or data.shape[0] < 60:
            skipped['record_too_short'] += 1
            continue

        n_rec = data.shape[0]
        drug_arrays = {t: data[:, j] for j, t in enumerate(drug_tracks)}
        sig = {k: data[:, len(drug_tracks) + j]
               for j, k in enumerate(aux_present.keys())}

        # BIS is meaningless below its signal quality floor
        if 'bis' in sig and 'bis_sqi' in sig:
            bis = sig['bis'].copy()
            bis[np.nan_to_num(sig['bis_sqi'], nan=0.0) < BIS_SQI_MIN] = np.nan
            sig['bis'] = bis

        # ---- anesthesia window, derived from the file alone -----------
        ane_start, ane_end, start_src, end_src, n_src = \
            derive_anesthesia_window(sig, n_rec)

        # ---- detect events across all pressor tracks ------------------
        case_events = []
        for track_name in drug_tracks:
            key = next((k for k in MIN_RATE_BY_DRUG if k in track_name.upper()),
                       None)
            min_rate = MIN_RATE_BY_DRUG.get(key, DEFAULT_MIN_RATE)
            values = drug_arrays[track_name]

            for s, e in detect_events(values, min_rate, GAP_MERGE_SEC,
                                      MIN_EVENT_SEC):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mx = float(np.nanmax(values[s:e + 1]))
                case_events.append({'track_name': track_name,
                                    'start_sec': s, 'end_sec': e,
                                    'max_rate': mx})

        if not case_events:
            continue

        case_events.sort(key=lambda d: (d['start_sec'], d['track_name']))

        for order, ev in enumerate(case_events, start=1):
            s, e = ev['start_sec'], ev['end_sec']

            concurrent = any(o is not ev and o['start_sec'] < s <= o['end_sec']
                             for o in case_events)

            rec = {
                # ---- original columns, unchanged ----
                'case_id': case_id,
                'track_name': ev['track_name'],
                'start_sec': s,
                'end_sec': e,
                'duration_sec': e - s,
                'max_rate': ev['max_rate'],
                # ---- derived anesthesia window ----
                'ane_start_est': ane_start,
                'ane_end_est': ane_end,
                'ane_start_src': start_src,
                'ane_end_src': end_src,
                'ane_n_sources': n_src,
                'ane_duration_sec': ane_end - ane_start,
                'rec_len_sec': n_rec,
                # ---- timing of the pressor within that window ----
                'anestart_to_pressor_sec': s - ane_start,
                'pressor_to_aneend_sec': ane_end - e,
                'ane_frac_at_onset': (s - ane_start) / max(1, ane_end - ane_start),
                'event_order': order,
                'n_events_in_case': len(case_events),
                'concurrent_pressor': concurrent,
                'pre_window_avail_sec': s,
                # ---- anesthetic identity ----
                'primary_anesthetic': classify_anesthetic(sig, s),
                'volatile_agent': volatile_agent(sig, s),
            }

            # ---- depth trajectory into the onset ----
            for label in ['mac', 'bis', 'ppf_ce', 'rftn20_ce', 'rftn50_ce']:
                arr = sig.get(label, EMPTY)
                for off in DEPTH_OFFSETS:
                    tag = f'm{abs(off)}' if off < 0 else 't0'
                    rec[f'{label}_{tag}'] = window_median(arr, s + off,
                                                          DEPTH_HALFWIN)
                rec[f'{label}_cov_pre900'] = coverage(arr, s - PREWIN_SEC, s)

            # ---- volatile concentration at onset ----
            for label in ['exp_sevo', 'exp_des', 'fin2o', 'gas2_exp']:
                rec[f'{label}_t0'] = window_median(sig.get(label, EMPTY), s,
                                                   DEPTH_HALFWIN)

            # ---- cumulative volatile exposure, ane_start -> onset ----
            mac = sig.get('mac', EMPTY)
            span = max(1, s - ane_start)
            if mac.size > 1 and s > ane_start:
                seg = mac[max(0, ane_start):s]
                valid = seg[~np.isnan(seg)]
                rec['mac_auc_pre'] = float(np.sum(valid))          # MAC-seconds
                rec['mac_twa_pre'] = float(np.mean(valid)) if valid.size else np.nan
                rec['mac_cov_pre'] = float(valid.size / span)
            else:
                rec['mac_auc_pre'] = np.nan
                rec['mac_twa_pre'] = np.nan
                rec['mac_cov_pre'] = 0.0

            # ---- dose delivered in the 15-min pre-window ----
            lo = s - PREWIN_SEC
            ppf_ml = cumulative_delta(sig.get('ppf_vol', EMPTY), lo, s)
            r20 = cumulative_delta(sig.get('rftn20_vol', EMPTY), lo, s)
            r50 = cumulative_delta(sig.get('rftn50_vol', EMPTY), lo, s)

            rec['ppf_mg_pre900'] = ppf_ml * 20 if np.isfinite(ppf_ml) else np.nan
            if np.isfinite(r20) or np.isfinite(r50):
                rec['rftn_mcg_pre900'] = float(np.nansum([
                    r20 * 20 if np.isfinite(r20) else np.nan,
                    r50 * 50 if np.isfinite(r50) else np.nan]))
            else:
                rec['rftn_mcg_pre900'] = np.nan

            dex = [window_median(sig.get('dex2_rate', EMPTY), s, DEPTH_HALFWIN),
                   window_median(sig.get('dex4_rate', EMPTY), s, DEPTH_HALFWIN)]
            dex = [d for d in dex if np.isfinite(d)]
            rec['dex_rate_t0'] = max(dex) if dex else np.nan

            all_events.append(rec)

        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)} | {len(all_events)} events "
                  f"| {(i+1)/el:.1f} files/s")

    # ==========================================
    # OUTPUT
    # ==========================================
    df = pd.DataFrame(all_events)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 46)
    n_cases = df['case_id'].nunique() if len(df) else 0
    print(f"events: {len(df)} | cases: {n_cases}")

    if len(df):
        print("\nanesthesia-window source (start):")
        print(df['ane_start_src'].value_counts().to_string())
        print(f"\nrecord_fallback cases (no agent/ETCO2 track): "
              f"{(df['ane_start_src'] == 'record_fallback').sum()}")
        print("\nprimary_anesthetic:")
        print(df['primary_anesthetic'].value_counts().to_string())
        print(f"\nconcurrent_pressor events: {int(df['concurrent_pressor'].sum())}")
        print("\npressor_to_aneend_sec (min) -- for manual emergence cut:")
        print((df['pressor_to_aneend_sec'] / 60).describe().round(1).to_string())
        print("\ndepth coverage in 15-min pre-window (>50% data):")
        for label in ['mac', 'bis', 'ppf_ce']:
            print(f"  {label:8s} {(df[f'{label}_cov_pre900'] > 0.5).mean():.1%}")
        print("\nanestart_to_pressor_sec (min):")
        print((df['anestart_to_pressor_sec'] / 60).describe().round(1).to_string())

    print("\nskipped:")
    for k, v in skipped.most_common():
        print(f"  {k:24s} {v}")
    if failed:
        print(f"\nfailures: {len(failed)}")
        for cid, kind, msg in failed[:10]:
            print(f"  {cid} {kind} {msg[:80]}")
    print(f"\nsaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()