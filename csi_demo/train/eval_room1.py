#!/usr/bin/env python3
"""Does the model's accuracy come from the CSI? Strict 2D evaluation on a held-out session.

    python csi_demo/train/eval_room1.py CONFIG CKPT [CKPT ...] --cache data/cache_room1

For each checkpoint it scores only the single highest-confidence pose per frame (the official
pck2d matches every GT to the closest of all 50 queries, which rewards guessing many poses), and
compares with no-CSI baselines computed from the training labels:
  mean pose     - the average training pose every frame
  random pose   - a random training pose every frame
Errors are percent of image size; pck@0.1 = fraction of joints within 10% of the image.
Reported for all joints and for upper body (head, neck, shoulders, elbows, hands, hips), since the
camera cannot see the knees/ankles in the current recordings.
"""
import argparse

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from opera.models import build_model

UPPER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]


def score(pred, gt, joints):
    d = np.linalg.norm(pred[:, joints] - gt[:, joints], axis=-1)
    return f'err {100 * d.mean():5.2f}%  pck@0.05 {np.mean(d < 0.05):.3f}  pck@0.1 {np.mean(d < 0.1):.3f}'


def report(name, pred, gt):
    print(f'{name:28s} all: {score(pred, gt, range(14))} | upper: {score(pred, gt, UPPER)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--cache', default='data/cache_room1')
    args = ap.parse_args()

    tr_kp = np.load(f'{args.cache}/train/keypoints.npy')[:, 0]
    te_csi = np.load(f'{args.cache}/test/csi.npy')
    gt = np.load(f'{args.cache}/test/keypoints.npy')[:, 0]
    print(f'test frames {len(gt)}, train frames {len(tr_kp)}')
    report('baseline: mean pose', np.repeat(tr_kp.mean(0, keepdims=True), len(gt), 0), gt)
    rng = np.random.default_rng(0)
    report('baseline: random train pose', tr_kp[rng.integers(0, len(tr_kp), len(gt))], gt)

    cfg = Config.fromfile(args.config)
    for ckpt in args.ckpts:
        model = build_model(cfg.model)
        load_checkpoint(model, ckpt, map_location='cpu')
        model = model.cuda().eval()
        pred = np.zeros_like(gt)
        with torch.no_grad():
            for i in range(0, len(te_csi)):
                bboxes, kpts = model.simple_test(torch.from_numpy(te_csi[i:i + 1]).cuda(), [dict()])[0]
                pred[i] = kpts[0][int(np.argmax(bboxes[0][:, -1]))][:, :2]
        report(ckpt.split('work_dirs/')[-1], pred, gt)


if __name__ == '__main__':
    main()
