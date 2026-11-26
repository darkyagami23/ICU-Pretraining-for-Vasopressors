"""
check_checkpoint.py -- STEP 0. Tells you what an encoder checkpoint really contains.

Usage (Kaggle cell):
    !python check_checkpoint.py /kaggle/input/.../hybrid_v4_encoders.pth
    !python check_checkpoint.py /kaggle/input/.../ablation_abp_ecg_encoders.pth

It answers:
  * Is the ECG branch the wavelet encoder or a raw-patch encoder?
  * Which patch length / number of patches was used (25/150 or 30/125)?
  * Would loading it into the downstream model leave any branch randomly
    initialised (the silent failure that strict=False hides)?
"""
import sys
import torch  # noqa: F401  (needed by common)
from common import read_state_dict, describe_checkpoint, build_encoder_bundle, load_encoder_checkpoint


def main(paths):
    for path in paths:
        print('=' * 70)
        print(path)
        state = read_state_dict(path)
        info = describe_checkpoint(state)
        for k, v in info.items():
            print(f'  {k:14s}: {v}')

        if info['format'] == 'trimodal':
            ecg = info['enc_ecg']
            pl = info['enc_ppg']['patch_len']
            print()
            if ecg['type'] != 'wavelet':
                print('  !! ECG branch is NOT the wavelet encoder. If the downstream notebook built '
                      'PrivateEncoderECG and loaded this with strict=False, the ECG front-end was '
                      'randomly initialised at fine-tuning.')
            else:
                print('  OK: ECG branch is the learnable-wavelet encoder.')
            print(f'  Patch length = {pl}  ->  {3750 // pl if pl else "?"} patches per modality '
                  f'(paper currently says 25 / 150).')
            # Simulate the downstream load
            bundle = build_encoder_bundle(patch_len=pl or 30, ecg_encoder='wavelet')
            missing, unexpected = load_encoder_checkpoint(bundle, path, verbose=False)
            print(f'  Loading into the downstream (wavelet-ECG) model: missing={len(missing)}, '
                  f'unexpected={len(unexpected)}')
            if missing:
                print('    first missing keys:', missing[:5])
        else:
            print('  Pair checkpoint: branches enc_a / enc_b (order is in the file name).')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python check_checkpoint.py <encoders.pth> [more.pth ...]')
    main(sys.argv[1:])
