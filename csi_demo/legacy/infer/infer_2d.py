#!/usr/bin/env python3
"""Run the 2D Wi-Fi pose model on CSI windows and print the detected 2D poses.

    python csi_demo/infer/infer_2d.py configs/wifi/petr_wifi_esp_like_2d_150e.py CHECKPOINT windows.npz \
        [--score 0.3] [--every 15] [--out poses.npz] [--draw DIR]

Keypoints are in the model's normalized [0,1] image coordinates (x right, y down).
"""
import argparse

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from opera.models import build_model

NAMES = ['neck', 'head', 'l_shoulder', 'r_shoulder', 'l_elbow', 'l_hip', 'r_elbow', 'r_hip',
         'l_hand', 'l_knee', 'r_hand', 'r_knee', 'l_ankle', 'r_ankle']
EDGES = [(0, 1), (0, 2), (0, 3), (2, 4), (4, 8), (3, 6), (6, 10), (2, 5), (3, 7), (5, 7),
         (5, 9), (9, 12), (7, 11), (11, 13)]


def draw(path, people, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    for kp, score in people:
        for a, b in EDGES:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]], '-', lw=2)
        ax.scatter(kp[:, 0], kp[:, 1], s=12)
        ax.text(kp[1, 0], kp[1, 1] - 0.03, f'{score:.2f}', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('windows')
    ap.add_argument('--score', type=float, default=0.3)
    ap.add_argument('--every', type=int, default=15, help='print every N-th window')
    ap.add_argument('--out')
    ap.add_argument('--draw')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    z = np.load(args.windows)
    csi, times = z['csi'], z['t']
    print(f'{len(csi)} windows of shape {csi.shape[1:]}, boards {list(z["boards"])}')
    all_kpts, all_scores = [], []
    for i, win in enumerate(csi):
        with torch.no_grad():
            bbox_results, kpt_results = model.simple_test(torch.from_numpy(win[None]).cuda(), [dict()])[0]
        scores, kpts = bbox_results[0][:, -1], kpt_results[0]
        keep = scores >= args.score
        all_kpts.append(kpts)
        all_scores.append(scores)
        if args.draw and i % args.every == 0:
            import os
            os.makedirs(args.draw, exist_ok=True)
            draw(f'{args.draw}/win_{i:05d}.png', list(zip(kpts[keep], scores[keep])), f't={times[i] - times[0]:.2f}s')
        if i % args.every == 0:
            print(f'--- window {i} t={times[i] - times[0]:.2f}s: {int(keep.sum())} person(s) with score >= {args.score} '
                  f'(top score {scores.max():.3f})')
            for p, (kp, s) in enumerate(zip(kpts[keep][:3], scores[keep][:3])):
                print(f'  person {p} score {s:.3f}: ' + ', '.join(
                    f'{n}=({x:.3f},{y:.3f})' for n, (x, y) in zip(NAMES, kp[:, :2])))
    if args.out:
        np.savez(args.out, t=times, scores=np.array(all_scores, dtype=object),
                 keypoints=np.array(all_kpts, dtype=object), allow_pickle=True)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
