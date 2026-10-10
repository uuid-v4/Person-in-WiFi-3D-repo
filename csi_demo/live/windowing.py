"""Build model input windows from raw CSI records exactly like dataset/build_2d_cache.py.

A window for time T takes, for every (receiver, transmitter) link, the last 20 records with Mac time <= T
(oldest no older than T - max_age), turns each record into 30 amplitudes (+30 sanitized phases for v2),
and applies the builder's smooth_window. Result: float32 (n_rx, n_tx, 20, 30|60), or None if any link
lacks enough recent packets.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset'))
import build_2d_cache as B  # noqa: E402

MODES = {'v1': dict(phase=False, packet_norm=False), 'v2': dict(phase=True, packet_norm=True)}


def record_features(csi, phase):
    """csi: (R, 104) int8 records -> (R, 30) or (R, 60) float32, same math as build_2d_cache.load_board."""
    iq = np.asarray(csi, np.float32).reshape(-1, 52, 2)
    amp = np.hypot(iq[..., 1], iq[..., 0])
    amp = np.concatenate([amp[:, 26:], amp[:, :26]], axis=1)
    feat = np.stack([np.interp(B.TARGET_SUB, B.SUB_ORDER, a) for a in amp])
    if phase:
        feat = np.concatenate([feat, B.sanitized_phase(iq)], axis=1)
    return feat.astype(np.float32)


def build_window(links, boards, tx_ids, T, mode='v1', max_age=0.7):
    """links: {(board, tx): (times (R,) sorted, csi (R, 104) int8)}."""
    opts = MODES[mode]
    B.OPTS.update(opts)
    out = []
    for b in boards:
        for t in tx_ids:
            times, csi = links.get((b, t), (np.zeros(0), np.zeros((0, 104), np.int8)))
            end = int(np.searchsorted(times, T, side='right'))
            if end < B.WIN or times[end - B.WIN] < T - max_age:
                return None
            out.append(B.smooth_window(record_features(csi[end - B.WIN:end], opts['phase'])))
    return np.stack(out).reshape(len(boards), len(tx_ids), B.WIN, -1).astype(np.float32)
