#!/usr/bin/env python3
"""Turn BLE CSI uploads into model input windows at a fixed rate (no camera labels needed).

    python csi_demo/infer/make_windows.py --csi-root recordings --boards CSI-A,CSI-B,CSI-C --out windows.npz [--hz 15]

Uses the same per-board loading, 30-subcarrier mapping and smoothing as dataset/build_2d_cache.py,
so windows match the training cache layout: csi (N, n_boards, 1, 20, 30), plus t (N,) Mac times.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset'))
from build_2d_cache import MAX_AGE_S, WIN, load_board, smooth_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csi-root', required=True)
    ap.add_argument('--boards', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hz', type=float, default=15.0)
    args = ap.parse_args()

    boards = args.boards.split(',')
    data = [load_board(Path(args.csi_root) / b) for b in boards]
    for b, (t, _) in zip(boards, data):
        rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0
        print(f'{b}: {len(t)} records, {t[0]:.2f}..{t[-1]:.2f} ({rate:.1f} Hz)')
    start = max(t[WIN - 1] for t, _ in data)
    end = min(t[-1] for t, _ in data)
    if end <= start:
        raise SystemExit('boards have no overlapping time span with >= 20 records each')

    wins, times = [], []
    for T in np.arange(start, end, 1.0 / args.hz):
        per_board = []
        for t, amp in data:
            i = np.searchsorted(t, T, side='right')
            if i < WIN or T - t[i - WIN] > MAX_AGE_S:
                break
            per_board.append(smooth_window(amp[i - WIN:i]))
        if len(per_board) == len(data):
            wins.append(np.stack(per_board)[:, None])
            times.append(T)
    csi = np.stack(wins).astype(np.float32)
    np.savez(args.out, csi=csi, t=np.array(times), boards=np.array(boards))
    print(f'windows: {csi.shape} over {times[-1] - times[0]:.1f} s -> {args.out}')


if __name__ == '__main__':
    main()
