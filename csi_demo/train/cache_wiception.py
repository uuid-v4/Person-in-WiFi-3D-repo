#!/usr/bin/env python3
"""Precompute Wiception3D CSI features once so training is GPU-bound instead of loader-bound.

Applies exactly WifiPoseDataset's per-sample processing (DWT amplitude + sanitized phase,
shape 3x3x20x60) and writes memory-mappable arrays for WifiPoseCachedDataset:

    python csi_demo/train/cache_wiception.py --root data/wiception3d/train_data --split train \
        --out data/wiception3d_cache/train --workers 16
"""
import argparse
import os
from multiprocessing import Pool

import h5py
import numpy as np

from opera.datasets.wifi_pose import WifiPoseDataset

PROC = WifiPoseDataset.__new__(WifiPoseDataset)


def read_names(list_file):
    with open(list_file) as f:
        return [line.split()[0] for line in f if line.strip()]


def process(job):
    root, name = job
    with h5py.File(os.path.join(root, 'csi', f'{name}.mat'), 'r') as f:
        raw = f['csi_out'][()]
    csi = (raw['real'] + raw['imag'] * 1j).transpose(3, 2, 1, 0).astype(np.complex128)
    amp = PROC.dwt_amp(csi)
    phase = np.angle(PROC.phase_deno(csi))
    feat = np.concatenate((amp, phase), axis=2).transpose(0, 1, 3, 2).astype(np.float32)
    kpt = np.load(os.path.join(root, 'keypoint', f'{name}.npy')).astype(np.float32)
    return feat, kpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='split directory containing csi/, keypoint/, <split>_data_list.txt')
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    names = read_names(os.path.join(args.root, f'{args.split}_data_list.txt'))
    os.makedirs(args.out, exist_ok=True)
    first, _ = process((args.root, names[0]))
    csi = np.lib.format.open_memmap(os.path.join(args.out, 'csi.npy'), mode='w+',
                                    dtype=np.float32, shape=(len(names),) + first.shape)
    keypoints = []
    with Pool(args.workers) as pool:
        for i, (feat, kpt) in enumerate(pool.imap(process, [(args.root, n) for n in names], chunksize=64)):
            csi[i] = feat
            keypoints.append(kpt)
            if i % 5000 == 0:
                print(f'{args.split}: {i}/{len(names)}', flush=True)
    csi.flush()

    max_p = max(k.shape[0] for k in keypoints)
    padded = np.zeros((len(names), max_p) + keypoints[0].shape[1:], dtype=np.float32)
    for i, k in enumerate(keypoints):
        padded[i, :k.shape[0]] = k
    np.save(os.path.join(args.out, 'keypoints.npy'), padded)
    np.save(os.path.join(args.out, 'num_person.npy'), np.array([k.shape[0] for k in keypoints], dtype=np.int16))
    np.save(os.path.join(args.out, 'names.npy'), np.array(names))
    print(f'{args.split}: done, {len(names)} samples, csi {csi.shape}, keypoints {padded.shape}')


if __name__ == '__main__':
    main()
