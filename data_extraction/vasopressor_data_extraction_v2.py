import os
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
import vitaldb
from tqdm import tqdm
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MANIFEST_PATH   = r"D:\vasopressor_labels_concise.csv"          # new labels (onset >= 20 min after ane start)
TRIPLE_JSON     = r"D:\valid_triple_track_cases.json"            # case-level PPG+ECG+ABP whitelist
LOCAL_VITAL_DIR = r"D:\vitaldb\vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0\vital_files"
OUTPUT_DIR      = r"F:\Vasopressor_Processed_Data_Raw_v2"        # NEW folder -- don't mix with old run
SUMMARY_PATH    = os.path.join(OUTPUT_DIR, "extraction_summary.csv")

for sub in ['ppg', 'ecg', 'abp']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# Standardization parameters (unchanged)
FS_RAW              = 500
FS_TARGET           = 125
WINDOW_MINS         = 15
TOTAL_SEC           = WINDOW_MINS * 60
OFFSET_SEC          = 30                        # gap before drug onset
SEGMENT_SEC         = 30
NUM_SEGMENTS        = TOTAL_SEC // SEGMENT_SEC  # 30
SAMPLES_PER_SEGMENT = SEGMENT_SEC * FS_TARGET   # 3750

GAP_LIMIT_SEC         = 1.0
GAP_LIMIT_SAMPLES_RAW = int(GAP_LIMIT_SEC * FS_RAW)

TRACKS_WAVE = {
    'ppg': 'SNUADC/PLETH',
    'ecg': 'SNUADC/ECG_II',
    'abp': 'SNUADC/ART'
}

TRACKS_BASED_RAW = {
    'ppg': ['Solar8000/PLETH_SPO2', 'Solar8000/PLETH_HR'],
    'ecg': ['Solar8000/HR'],
    'abp': ['Solar8000/ART_SBP', 'Solar8000/ART_DBP', 'Solar8000/ART_MBP',
            'EV1000/SV',  'EV1000/SVV',  'EV1000/CO',
            'Vigileo/SV', 'Vigileo/SVV', 'Vigileo/CO']
}
TRACKS_BASED_FINAL = {
    'ppg': ['Solar8000/PLETH_SPO2', 'Solar8000/PLETH_HR'],
    'ecg': ['Solar8000/HR'],
    'abp': ['Solar8000/ART_SBP', 'Solar8000/ART_DBP', 'Solar8000/ART_MBP',
            'Merged_SV', 'Merged_SVV', 'Merged_CO']
}

DRUG_MAPPING = {'PHEN': 1, 'VASO': 1, 'NEPI': 2, 'EPI': 2, 'DOPA': 2}

BASE_COLS = ['case_id', 'label', 'label_binary', 'vasopressor_name',
             'segment_start_sec', 'time_offset']

# ==========================================
# 2. SIGNAL PROCESSING HELPERS (unchanged, no z-norm)
# ==========================================
def apply_bandpass_filter(data, lowcut, highcut, fs=FS_RAW, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band', analog=False)
    return filtfilt(b, a, data)

def apply_lowpass_filter(data, cutoff=25.0, fs=FS_RAW, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low', analog=False)
    return filtfilt(b, a, data)

def fill_gaps_and_zeropad(signal_array):
    s = pd.Series(signal_array)
    s = s.interpolate(method='pchip', limit=GAP_LIMIT_SAMPLES_RAW, limit_direction='forward')
    return s.fillna(0.0).values

def extract_15_points(data_segment, target_len=15):
    clean = data_segment[~np.isnan(data_segment)]
    if len(clean) < 2: return [np.nan] * target_len
    idx = np.linspace(0, len(clean)-1, target_len).astype(int)
    return clean[idx].tolist()

def merge_ev1000_vigileo(df):
    for param in ['SV', 'SVV', 'CO']:
        ev, vg, mg = f'EV1000/{param}', f'Vigileo/{param}', f'Merged_{param}'
        if ev in df.columns and vg in df.columns: df[mg] = df[ev].combine_first(df[vg])
        elif ev in df.columns:                   df[mg] = df[ev]
        elif vg in df.columns:                   df[mg] = df[vg]
        else:                                    df[mg] = np.nan
    return df

def sensor_present_check(ppg, ecg, abp, max_nan_frac=0.10):
    """Reject only if a sensor was disconnected (>max_nan_frac NaN)."""
    if np.isnan(abp).mean() > max_nan_frac: return False
    if np.isnan(ecg).mean() > max_nan_frac: return False
    if np.isnan(ppg).mean() > max_nan_frac: return False
    return True

def find_vital_file(case_id):
    for name in (f"{case_id:04d}.vital", f"Case_{case_id}.vital"):
        p = os.path.join(LOCAL_VITAL_DIR, name)
        if os.path.exists(p): return p
    return None

# ==========================================
# 3. ONE DOSE -> 30 SEGMENTS (same logic as before)
# ==========================================
def process_one_dose(case_id, dose_index, event, wave_full, df_based_all):
    onset_sec = float(event['start_sec'])
    drug_name = str(event['track_name']).split('/')[-1].replace('_RATE', '')
    vp_label  = DRUG_MAPPING.get(drug_name.split('_')[0], 1)

    # 15-min window ending 30 s before onset
    end_sec_window   = onset_sec - OFFSET_SEC
    start_sec_window = end_sec_window - TOTAL_SEC
    if start_sec_window < 0:
        return 'window_negative'

    idx_s, idx_e = int(start_sec_window * FS_RAW), int(end_sec_window * FS_RAW)
    if idx_e > len(wave_full):
        return 'window_oob'

    ppg_raw = wave_full[idx_s:idx_e, 0]
    ecg_raw = wave_full[idx_s:idx_e, 1]
    abp_raw = wave_full[idx_s:idx_e, 2]

    if not sensor_present_check(ppg_raw, ecg_raw, abp_raw):
        return 'sensor_disconnected'

    # --- preprocess (no z-norm) ---
    waves_125hz = {}
    ppg_f = apply_bandpass_filter(fill_gaps_and_zeropad(ppg_raw), 0.5, 25.0)
    waves_125hz['ppg'] = resample_poly(ppg_f, FS_TARGET, FS_RAW)

    ecg_f = apply_bandpass_filter(fill_gaps_and_zeropad(np.clip(ecg_raw, -5.0, 5.0)), 0.5, 40.0)
    waves_125hz['ecg'] = resample_poly(ecg_f, FS_TARGET, FS_RAW)

    abp_f = apply_lowpass_filter(fill_gaps_and_zeropad(np.clip(abp_raw, 0.0, 350.0)), 25.0)
    waves_125hz['abp'] = resample_poly(abp_f, FS_TARGET, FS_RAW)

    # --- 30 segments, chronological ---
    datasets = {f"{m}_{dt}": [] for m in ['ppg', 'ecg', 'abp'] for dt in ['wave', 'based']}
    for seg_idx in range(NUM_SEGMENTS):
        t_start_rel = seg_idx * SEGMENT_SEC
        t_end_rel   = t_start_rel + SEGMENT_SEC
        t_start_abs = start_sec_window + t_start_rel
        t_end_abs   = start_sec_window + t_end_rel

        sec_to_onset = int(round(onset_sec - t_end_abs))
        rel_time     = f"-{sec_to_onset//60}m{sec_to_onset%60:02d}s"
        meta = [case_id, vp_label, 1, drug_name, round(t_start_abs, 2), rel_time]

        for mod in ['ppg', 'ecg', 'abp']:
            s, e = int(t_start_rel * FS_TARGET), int(t_end_rel * FS_TARGET)
            seg_wave = waves_125hz[mod][s:e]
            if len(seg_wave) != SAMPLES_PER_SEGMENT:
                seg_wave = np.pad(seg_wave, (0, max(0, SAMPLES_PER_SEGMENT - len(seg_wave))),
                                  'edge')[:SAMPLES_PER_SEGMENT]
            datasets[f"{mod}_wave"].append(meta + seg_wave.tolist())

        for mod in ['ppg', 'ecg', 'abp']:
            row_based = []
            for t in TRACKS_BASED_FINAL[mod]:
                if t in df_based_all.columns:
                    d = df_based_all[(df_based_all.index >= t_start_abs) &
                                     (df_based_all.index <  t_end_abs)][t].values
                    row_based.extend(extract_15_points(d))
                else:
                    row_based.extend([np.nan] * 15)
            datasets[f"{mod}_based"].append(meta + row_based)

    # --- save ---
    for mod in ['ppg', 'ecg', 'abp']:
        wave_cols  = BASE_COLS + [f"{mod}_{i}" for i in range(SAMPLES_PER_SEGMENT)]
        based_cols = BASE_COLS.copy()
        for t in TRACKS_BASED_FINAL[mod]:
            clean_name = t.split('/')[-1].lower()
            based_cols += [f"{clean_name}_{i}" for i in range(15)]

        stem = f"Case_{case_id:04d}_dose_{dose_index}_{mod}"
        pd.DataFrame(datasets[f"{mod}_wave"],  columns=wave_cols ).to_csv(
            os.path.join(OUTPUT_DIR, mod, f"{stem}_wave.csv"),  index=False)
        pd.DataFrame(datasets[f"{mod}_based"], columns=based_cols).to_csv(
            os.path.join(OUTPUT_DIR, mod, f"{stem}_based.csv"), index=False)

    return 'ok'

# ==========================================
# 4. MAIN LOOP  (one .vital load per case)
# ==========================================
def process_vp_data():
    print("VASOPRESSOR EXTRACTION  (concise labels + triple-track case whitelist, raw values)\n")

    # --- labels ---
    df = pd.read_csv(MANIFEST_PATH)
    df['case_id'] = df['case_id'].astype(int)
    df = df.sort_values(['case_id', 'start_sec']).reset_index(drop=True)
    df['dose_index'] = df.groupby('case_id').cumcount() + 1     # numbering within concise CSV
    n_events, n_cases = len(df), df.case_id.nunique()

    # --- case-level whitelist ---
    with open(TRIPLE_JSON) as f:
        triple = set(int(c) for c in json.load(f))
    df = df[df.case_id.isin(triple)]

    print(f"labels         : {n_events} events / {n_cases} cases")
    print(f"triple-track   : {len(df)} events / {df.case_id.nunique()} cases\n")

    skipped = defaultdict(int)
    summary = []

    for case_id, events in tqdm(df.groupby('case_id'), desc="cases"):
        path = find_vital_file(case_id)
        if path is None:
            skipped['file_missing'] += len(events)
            continue

        try:
            vf = vitaldb.VitalFile(path)
            existing = vf.get_track_names()
            if not all(TRACKS_WAVE[m] in existing for m in ['ppg', 'ecg', 'abp']):
                skipped['modality_missing_unexpected'] += len(events)
                continue

            # all three waves at once -> same length, aligned  (cols: ppg, ecg, abp)
            wave_full = vf.to_numpy([TRACKS_WAVE['ppg'], TRACKS_WAVE['ecg'], TRACKS_WAVE['abp']],
                                    interval=1/FS_RAW)

            df_based_all = pd.DataFrame()
            avail = [t for t in sum(TRACKS_BASED_RAW.values(), []) if t in existing]
            if avail:
                df_based_all = vf.to_pandas(avail, interval=1).ffill().bfill()
                df_based_all = merge_ev1000_vigileo(df_based_all)
        except Exception as e:
            print(f"  [!] case {case_id}: load failed -> {e}")
            skipped['exception'] += len(events)
            continue

        for _, ev in events.iterrows():
            dose_index = int(ev['dose_index'])
            try:
                status = process_one_dose(case_id, dose_index, ev, wave_full, df_based_all)
            except Exception as e:
                print(f"  [!] case {case_id} dose {dose_index}: {e}")
                status = 'exception'

            if status != 'ok':
                skipped[status] += 1
            summary.append({'case_id': case_id, 'dose_index': dose_index,
                            'track_name': ev['track_name'], 'start_sec': ev['start_sec'],
                            'status': status})

    # --- report ---
    summ = pd.DataFrame(summary)
    summ.to_csv(SUMMARY_PATH, index=False)
    ok = summ[summ.status == 'ok'] if len(summ) else summ

    print(f"\nsuccessful doses : {len(ok)}  across {ok.case_id.nunique() if len(ok) else 0} cases")
    print("skipped:")
    for k, v in skipped.items():
        print(f"  {k:28s} {v}")
    if len(ok):
        print("\ndrug breakdown (ok):")
        print(ok.track_name.value_counts().to_string())
    print(f"\nsummary saved -> {SUMMARY_PATH}")

process_vp_data()