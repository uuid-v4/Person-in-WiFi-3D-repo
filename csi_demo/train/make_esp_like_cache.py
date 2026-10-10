#!/usr/bin/env python3
"""Derive an ESP32-like Wiception3D cache from the full cache made by cache_wiception.py.

Keeps one antenna per receiver and amplitude only, then normalizes each receiver's
(time x subcarrier) window by its mean so inputs are gain/scale independent:
(N, 3 rx, 3 ant, 20, 60 [amp|phase]) -> (N, 3 rx, 1 ant, 20, 30).

    python csi_demo/train/make_esp_like_cache.py --src data/wiception3d_cache --dst data/wiception3d_cache_esp
"""
import argparse
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--antenna', type=int, default=0)
    args = ap.parse_args()

    for split in ('train', 'test'):
        src, dst = os.path.join(args.src, split), os.path.join(args.dst, split)
        os.makedirs(dst, exist_ok=True)
        csi = np.load(os.path.join(src, 'csi.npy'), mmap_mode='r')
        out = np.lib.format.open_memmap(os.path.join(dst, 'csi.npy'), mode='w+', dtype=np.float32,
                                        shape=(csi.shape[0], csi.shape[1], 1, csi.shape[3], 30))
        for start in range(0, csi.shape[0], 4096):
            amp = np.asarray(csi[start:start + 4096, :, args.antenna, :, :30], dtype=np.float32)
            amp /= amp.mean(axis=(2, 3), keepdims=True) + 1e-6
            out[start:start + 4096] = amp[:, :, None]
        out.flush()
        for name in ('keypoints.npy', 'num_person.npy', 'names.npy'):
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
        print(f'{split}: {out.shape}, mean {float(out[:2000].mean()):.3f}, std {float(out[:2000].std()):.3f}')


if __name__ == '__main__':
    main()
