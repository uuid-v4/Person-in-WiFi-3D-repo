#!/usr/bin/env python3
"""Run a trained checkpoint on a dataset cache split and save the most confident 2D pose per window.

    PYTHONPATH=. python csi_demo/infer/predict.py CONFIG CKPT --cache data/cache_9link --split test \
        [--start 0 --count 0] --out preds.npz

Output npz: pred (N,14,2) normalized image coords, score (N,), names (N,) '<session>_<frame>', start.
--count 0 = the whole split. Uses the GPU if available.
"""
import argparse

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from opera.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    csi = np.load(f'{args.cache}/{args.split}/csi.npy', mmap_mode='r')
    names = np.load(f'{args.cache}/{args.split}/names.npy')
    end = len(csi) if args.count <= 0 else min(len(csi), args.start + args.count)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(Config.fromfile(args.config).model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(dev).eval()
    preds, scores = [], []
    with torch.no_grad():
        for w in csi[args.start:end]:
            bboxes, kpts = model.simple_test(torch.from_numpy(np.ascontiguousarray(w[None])).to(dev), [dict()])[0]
            i = int(np.argmax(bboxes[0][:, -1]))
            preds.append(kpts[0][i][:, :2])
            scores.append(float(bboxes[0][i, -1]))
    np.savez(args.out, pred=np.array(preds, np.float32), score=np.array(scores, np.float32),
             names=names[args.start:end], start=args.start)
    print(f'{len(preds)} windows -> {args.out}')


if __name__ == '__main__':
    main()
