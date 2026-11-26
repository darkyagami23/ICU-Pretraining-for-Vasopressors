"""
common.py -- shared code for the JBHI experiments.

Architecture is copied from your notebooks (downstream_code.ipynb Cell 1 and
pretraining-code.ipynb Cell 1) with three additions that do NOT change any
parameter names, so old checkpoints still load:

  1. SharedCrossModalEncoder.forward_subset(...) runs the shared transformer
     on any subset of modalities (for single-/two-modality fine-tuning).
  2. build_encoder_bundle(...) can build the ECG branch as the wavelet encoder
     (default) or as a raw-patch encoder (ablation A6).
  3. load_encoder_checkpoint(...) loads a tri-modal bundle checkpoint OR one of
     your pair checkpoints (ablation_<a>_<b>_encoders.pth, keys enc_a/enc_b),
     and REPORTS missing/unexpected keys loudly instead of silently.

Also: dataset (unchanged robust-z logic), window metadata parsing, and
patient-level subset helpers.
"""
import os
import re
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_utils import (parse_filename, robust_z, window_table, patient_subset,  # noqa: F401
                        save_json)

CANONICAL = ['ppg', 'abp', 'ecg']           # channel order in the memmaps / model
MOD_ID = {'ppg': 0, 'abp': 1, 'ecg': 2}     # modality_embed ids (same as notebooks)
SEQ_LEN = 3750
FS = 125


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Architecture (verbatim from the notebooks unless marked NEW)
# =============================================================================
DB6_DEC_LO = torch.tensor([
    -0.00107730108499558, 0.00477725751101065, 0.000553842200993802,
    -0.031582039318031156, 0.027522865530016288, 0.09750160558707936,
    -0.12976686756709563, -0.22626469396516913, 0.31525035170924843,
    0.7511339080215775, 0.4946238903983854, 0.11154074335008017,
], dtype=torch.float32)


def qmf(lo):
    n = lo.shape[-1]
    sign = torch.tensor([(-1.0) ** i for i in range(n)], dtype=lo.dtype, device=lo.device)
    return sign * lo.flip(-1)


def _conv_same(x, w, dilation=1):
    f = w.shape[-1]
    pad_total = (f - 1) * dilation
    pad_left = pad_total // 2
    return F.conv1d(F.pad(x, (pad_left, pad_total - pad_left)), w, dilation=dilation)


class LearnableWavelet1D(nn.Module):
    def __init__(self, n_levels=3, init_lo=DB6_DEC_LO):
        super().__init__()
        self.n_levels = n_levels
        self.filter_len = init_lo.shape[-1]
        init_hi = qmf(init_lo)
        self.lo = nn.ParameterList([nn.Parameter(init_lo.clone().view(1, 1, -1)) for _ in range(n_levels)])
        self.hi = nn.ParameterList([nn.Parameter(init_hi.clone().view(1, 1, -1)) for _ in range(n_levels)])

    @property
    def num_bands(self):
        return self.n_levels + 1

    def forward(self, x):
        x = x.unsqueeze(1)
        details, current = [], x
        for lvl in range(self.n_levels):
            d = 2 ** lvl
            lo_out = _conv_same(current, self.lo[lvl], dilation=d)
            hi_out = _conv_same(current, self.hi[lvl], dilation=d)
            details.append(hi_out)
            current = lo_out
        return torch.cat(details + [current], dim=1)

    def qmf_penalty(self):
        loss = 0.0
        for k in range(self.n_levels):
            loss = loss + F.mse_loss(self.hi[k].squeeze(), qmf(self.lo[k].squeeze()))
        return loss / self.n_levels


def _tf_layer(d_model, n_heads, dropout):
    return nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout,
        activation='gelu', batch_first=True, norm_first=True)


class PrivateEncoderRaw(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, patch_len=30, d_model=128, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.patch_len, self.d_model = patch_len, d_model
        self.num_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(_tf_layer(d_model, n_heads, dropout), num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        z = self.dropout(self.patch_proj(patches) + self.pos_embed)
        return self.norm(self.transformer(z))


class PrivateEncoderECG(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, patch_len=30, d_model=128, n_heads=8, n_layers=2,
                 dropout=0.1, n_wavelet_levels=3):
        super().__init__()
        self.patch_len, self.d_model = patch_len, d_model
        self.num_patches = seq_len // patch_len
        self.wavelet = LearnableWavelet1D(n_levels=n_wavelet_levels)
        self.K = self.wavelet.num_bands
        self.band_proj = nn.ModuleList([nn.Linear(patch_len, d_model) for _ in range(self.K)])
        self.band_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.K)])
        self.band_logits = nn.Parameter(torch.zeros(self.K))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(_tf_layer(d_model, n_heads, dropout), num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        bands = self.wavelet(x)
        patches = bands.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        proj = torch.stack([self.band_norm[k](self.band_proj[k](patches[:, k])) for k in range(self.K)], dim=1)
        w = F.softmax(self.band_logits, dim=0)
        z = (proj * w.view(1, -1, 1, 1)).sum(dim=1)
        z = self.dropout(z + self.pos_embed)
        return self.norm(self.transformer(z))


class SharedCrossModalEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=3, dropout=0.1, num_patches=125):
        super().__init__()
        self.d_model, self.num_patches = d_model, num_patches
        self.modality_embed = nn.Embedding(3, d_model)
        self.transformer = nn.TransformerEncoder(_tf_layer(d_model, n_heads, dropout), num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_ppg, z_abp, z_ecg):
        out = self.forward_subset([z_ppg, z_abp, z_ecg], ['ppg', 'abp', 'ecg'])
        return out[0], out[1], out[2]

    # NEW: any subset of modalities, same weights
    def forward_subset(self, z_list, mods):
        P = z_list[0].shape[1]
        dev = z_list[0].device
        toks = [z + self.modality_embed(torch.tensor([MOD_ID[m]], device=dev)).view(1, 1, -1)
                for z, m in zip(z_list, mods)]
        z = self.norm(self.transformer(torch.cat(toks, dim=1)))
        return [z[:, i * P:(i + 1) * P] for i in range(len(mods))]


class HybridEncoderBundle(nn.Module):
    """Same parameter names as the notebooks (enc_ppg, enc_abp, enc_ecg, shared)."""
    def __init__(self, enc_ppg, enc_abp, enc_ecg, shared):
        super().__init__()
        self.enc_ppg, self.enc_abp, self.enc_ecg, self.shared = enc_ppg, enc_abp, enc_ecg, shared

    def forward(self, x_ppg, x_abp, x_ecg):
        return self.shared(self.enc_ppg(x_ppg), self.enc_abp(x_abp), self.enc_ecg(x_ecg))

    def encode_subset(self, xs, mods):
        """xs: list of [B, L] tensors in the order of mods."""
        z = [getattr(self, f'enc_{m}')(x) for x, m in zip(xs, mods)]
        return self.shared.forward_subset(z, mods)


def build_encoder_bundle(patch_len=30, d_model=128, n_heads=8, private_layers=2, shared_layers=3,
                         n_wavelet_levels=3, dropout=0.1, ecg_encoder='wavelet', seq_len=SEQ_LEN):
    kw = dict(seq_len=seq_len, patch_len=patch_len, d_model=d_model, n_heads=n_heads,
              n_layers=private_layers, dropout=dropout)
    enc_ppg = PrivateEncoderRaw(**kw)
    enc_abp = PrivateEncoderRaw(**kw)
    if ecg_encoder == 'wavelet':
        enc_ecg = PrivateEncoderECG(n_wavelet_levels=n_wavelet_levels, **kw)
    elif ecg_encoder == 'raw':
        enc_ecg = PrivateEncoderRaw(**kw)
    else:
        raise ValueError(ecg_encoder)
    shared = SharedCrossModalEncoder(d_model=d_model, n_heads=n_heads, n_layers=shared_layers,
                                     dropout=dropout, num_patches=seq_len // patch_len)
    return HybridEncoderBundle(enc_ppg, enc_abp, enc_ecg, shared)


# =============================================================================
# Checkpoint inspection / loading
# =============================================================================
def _strip_prefix(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def read_state_dict(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    return state


def describe_checkpoint(state):
    """Infer what a bundle checkpoint contains. Works for tri-modal and pair files."""
    info = {}
    keys = list(state.keys())
    info['n_keys'] = len(keys)
    info['format'] = 'pair' if any(k.startswith('enc_a.') for k in keys) else 'trimodal'
    branches = ['enc_a', 'enc_b'] if info['format'] == 'pair' else ['enc_ppg', 'enc_abp', 'enc_ecg']
    for b in branches:
        has_wav = any(k.startswith(f'{b}.wavelet.') for k in keys)
        has_raw = f'{b}.patch_proj.weight' in state
        if has_wav:
            pl = state[f'{b}.band_proj.0.weight'].shape[1]
        elif has_raw:
            pl = state[f'{b}.patch_proj.weight'].shape[1]
        else:
            pl = None
        npatch = state[f'{b}.pos_embed'].shape[1] if f'{b}.pos_embed' in state else None
        info[b] = dict(type='wavelet' if has_wav else ('raw-patch' if has_raw else 'absent'),
                       patch_len=pl, num_patches=npatch)
    info['shared_layers'] = len({k.split('.')[3] for k in keys if k.startswith('shared.transformer.layers.')})
    return info


def load_encoder_checkpoint(bundle, path, pair_order=None, verbose=True):
    """Load encoder weights into a HybridEncoderBundle.

    Tri-modal checkpoint: keys enc_ppg.*, enc_abp.*, enc_ecg.*, shared.*
    Pair checkpoint (your ablation notebook): keys enc_a.*, enc_b.*, shared.*;
    pair_order=('abp','ecg') says which modality enc_a / enc_b are. If not
    given it is parsed from a file name like ablation_abp_ecg_encoders.pth.

    Returns (missing, unexpected). Raises if a branch the model needs got NO
    weights at all -- that is exactly the silent failure strict=False hides.
    """
    state = read_state_dict(path)
    if any(k.startswith('enc_a.') for k in state):
        if pair_order is None:
            m = re.search(r'ablation_(ppg|abp|ecg)_(ppg|abp|ecg)_', os.path.basename(path))
            if not m:
                raise ValueError('Pair checkpoint: pass pair_order=(mod_a, mod_b)')
            pair_order = (m.group(1), m.group(2))
        a, b = pair_order
        new = {}
        for k, v in state.items():
            if k.startswith('enc_a.'):
                new[f'enc_{a}.' + k[6:]] = v
            elif k.startswith('enc_b.'):
                new[f'enc_{b}.' + k[6:]] = v
            else:
                new[k] = v
        state = new
    missing, unexpected = bundle.load_state_dict(state, strict=False)
    if verbose:
        print(f'  [ckpt] {path}')
        print(f'  [ckpt] missing={len(missing)}  unexpected={len(unexpected)}')
        for k in missing[:8]:
            print(f'         missing    {k}')
        for k in unexpected[:8]:
            print(f'         unexpected {k}')
    return missing, unexpected


# =============================================================================
# Data
# =============================================================================
def mask_edges(wave, mask_start=0, mask_end=0):
    """Set the first / last samples of every channel to 0 (= the window median after robust z).
    Used to remove the start-of-window transient found in positive 15-min PPG windows."""
    if mask_start or mask_end:
        wave = wave.clone()
        if mask_start:
            wave[..., :mask_start] = 0
        if mask_end:
            wave[..., wave.shape[-1] - mask_end:] = 0
    return wave


class PhysiologicalDatasetMM(Dataset):
    """Same as the notebook's dataset; also exposes raw arrays and file names."""
    def __init__(self, memmap_dir, split, window, verbose=True, mask_start=0, mask_end=0):
        d = os.path.join(memmap_dir, split, window)
        self.mask_start, self.mask_end = int(mask_start), int(mask_end)
        N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
        self.N, self.split, self.window = N, split, window
        mm = lambda name, dt, shp: np.memmap(os.path.join(d, name), dtype=dt, mode='r', shape=shp)
        self.waves = {ch: mm(f'{ch}.dat', 'float32', (N, SEQ_LEN)) for ch in CANONICAL}
        self.labels = mm('labels.dat', 'int32', (N,))
        self.scalars = mm('scalars.dat', 'float32', (N, 3))
        self.map_tgts = mm('map_targets.dat', 'float32', (N, 15))
        self.sbp_tgts = mm('sbp_targets.dat', 'float32', (N, 15))
        self.dbp_tgts = mm('dbp_targets.dat', 'float32', (N, 15))
        fn_path = os.path.join(d, 'filenames.npy')
        self.filenames = np.load(fn_path, allow_pickle=True).astype(str) if os.path.exists(fn_path) else None
        if verbose:
            print(f'  [{split.upper()} | {window}] {N} samples  pos_rate={float(np.mean(self.labels)):.3f}')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        wave = torch.stack([torch.from_numpy(robust_z(np.array(self.waves[ch][idx]))) for ch in CANONICAL], 0)
        wave = mask_edges(wave, self.mask_start, self.mask_end)
        return ((wave, torch.from_numpy(np.array(self.scalars[idx], dtype=np.float32))),
                (torch.tensor(float(self.labels[idx])),
                 torch.from_numpy(np.array(self.map_tgts[idx], dtype=np.float32)),
                 torch.from_numpy(np.array(self.sbp_tgts[idx], dtype=np.float32)),
                 torch.from_numpy(np.array(self.dbp_tgts[idx], dtype=np.float32))))


