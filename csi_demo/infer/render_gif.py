#!/usr/bin/env python3
"""Overlay Wi-Fi predictions (red) and camera labels (green) on the session video and save a GIF.

    python csi_demo/infer/render_gif.py preds.npz --cache data/cache_9link --split test --out clip.gif \
        [--sessions data/sessions] [--every 2] [--max-frames 300]

preds.npz comes from predict.py; the window range must lie in one session (it is cut at the first change).
Prints the mean joint error of the rendered frames (percent of image size).
"""
import argparse

import cv2
import numpy as np
from PIL import Image

EDGES = [(0, 1), (0, 2), (0, 3), (2, 4), (4, 8), (3, 6), (6, 10), (2, 5), (3, 7), (5, 7), (5, 9), (9, 12),
         (7, 11), (11, 13)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preds')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--sessions', default='data/sessions')
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=300)
    args = ap.parse_args()

    z = np.load(args.preds)
    pred, names, start = z['pred'], z['names'], int(z['start'])
    sess = str(names[0]).split('_')[0]
    n = min(args.max_frames, next((i for i, s in enumerate(names) if str(s).split('_')[0] != sess), len(names)))
    gt = np.load(f'{args.cache}/{args.split}/keypoints.npy', mmap_mode='r')[start:start + n, 0]
    idx = [int(str(s).split('_')[1]) for s in names[:n]]
    cap = cv2.VideoCapture(f'{args.sessions}/{sess}/video.mp4')
    w, h = 320, 240
    frames = []
    for j in range(0, n, args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx[j])
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (w, h))
        blank = np.full((h, w, 3), 25, np.uint8)
        for canvas in (img, blank):
            for kp, col in ((gt[j], (60, 220, 60)), (pred[j], (60, 60, 255))):
                p = (np.asarray(kp) * [w, h]).astype(int)
                for a, b in EDGES:
                    cv2.line(canvas, tuple(p[a]), tuple(p[b]), col, 2, cv2.LINE_AA)
        err = 100 * np.linalg.norm(pred[j] - gt[j], axis=1).mean()
        cv2.putText(blank, 'green: camera  red: Wi-Fi', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(blank, f'{sess} frame {idx[j]}  err {err:4.1f}%', (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1)
        frames.append(Image.fromarray(cv2.cvtColor(np.hstack([img, blank]), cv2.COLOR_BGR2RGB)))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(43 * args.every), loop=0,
                   optimize=True)
    print(f'{len(frames)} frames -> {args.out}; mean joint error '
          f'{100 * np.linalg.norm(pred[:n] - gt, axis=2).mean():.2f}% of image')


if __name__ == '__main__':
    main()
