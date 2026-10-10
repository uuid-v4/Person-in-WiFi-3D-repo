#!/usr/bin/env python3
"""Build the ESP32-like Wiception3D cache directly from the Kaggle zip (no extraction, no opera import).

Per sample: |CSI| of antenna 0 for each receiver, smoothed with the same db11 wavelet
reconstruction as WifiPoseDataset.dwt_amp, laid out (3 rx, 1 ant, 20 time, 30 subcarriers),
then each receiver window divided by its mean. Matches make_esp_like_cache.py output.

    python csi_demo/train/esp_cache_from_zip.py --zip wiception3d.zip --out data/wiception3d_cache_esp --workers 8
"""
import argparse
import io
import os
import zipfile
from multiprocessing import Pool

import h5py
import numpy as np
import pywt

PREFIX = 'wifi_dataset'
_zip = None


def _open(path):
    global _zip
    _zip = zipfile.ZipFile(path)


def process(job):
    split, name = job
    with h5py.File(io.BytesIO(_zip.read(f'{PREFIX}/{split}_data/csi/{name}.mat')), 'r') as f:
        raw = f['csi_out'][()]
    csi = (raw['real'] + raw['imag'] * 1j).transpose(3, 2, 1, 0).astype(np.complex128)
    w = pywt.Wavelet('dB11')
    amp = pywt.waverec(pywt.wavedec(np.abs(csi[:, 0]), w, 'sym'), w)
    amp = amp.transpose(0, 2, 1)[:, None].astype(np.float32)
    amp /= amp.mean(axis=(2, 3), keepdims=True) + 1e-6
    kpt = np.load(io.BytesIO(_zip.read(f'{PREFIX}/{split}_data/keypoint/{name}.npy'))).astype(np.float32)
    return amp, kpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as z:
        lists = {s: [l.split()[0] for l in z.read(f'{PREFIX}/{s}_data/{s}_data_list.txt').decode().splitlines() if l.strip()]
                 for s in ('test', 'train')}
    with Pool(args.workers, initializer=_open, initargs=(args.zip,)) as pool:
        for split, names in lists.items():
            dst = os.path.join(args.out, split)
            os.makedirs(dst, exist_ok=True)
            csi = np.lib.format.open_memmap(os.path.join(dst, 'csi.npy'), mode='w+', dtype=np.float32,
                                            shape=(len(names), 3, 1, 20, 30))
            kpts = []
            for i, (amp, kpt) in enumerate(pool.imap(process, [(split, n) for n in names], chunksize=64)):
                csi[i] = amp
                kpts.append(kpt)
                if i % 10000 == 0:
                    print(f'{split}: {i}/{len(names)}', flush=True)
            csi.flush()
            max_p = max(k.shape[0] for k in kpts)
            padded = np.zeros((len(names), max_p) + kpts[0].shape[1:], dtype=np.float32)
            for i, k in enumerate(kpts):
                padded[i, :k.shape[0]] = k
            np.save(os.path.join(dst, 'keypoints.npy'), padded)
            np.save(os.path.join(dst, 'num_person.npy'), np.array([k.shape[0] for k in kpts], dtype=np.int16))
            np.save(os.path.join(dst, 'names.npy'), np.array(names))
            print(f'{split}: done {csi.shape}, persons {np.bincount([k.shape[0] for k in kpts])}', flush=True)


if __name__ == '__main__':
    main()
