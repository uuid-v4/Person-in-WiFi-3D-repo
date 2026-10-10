#!/usr/bin/env python3
"""Align ESP32 BLE CSI recordings with camera pose labels and write a 2D training cache.

    python build_2d_cache.py --boards CSI-A,CSI-B,CSI-C \
        --session recordings/s1:labels/s1 --session recordings/s2:labels/s2 \
        --out cache [--split-by-session s2] [--keep-empty]

CSI_ROOT contains one directory per board (collector.py output); LABELS_DIR contains labels.npz.
Output matches opera.WifiPoseCachedDataset: csi.npy (N, n_boards, 1, 20, 30), keypoints.npy
(N, P_max, 14, 2), num_person.npy, names.npy ('<session>_<frame index>'), meta.json.
With --split-by-session, DIR/train and DIR/test are written instead (each a full cache).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pywt

WIN = 20
MAX_AGE_S = 0.5
N_KPT = 14
SUB_ORDER = np.r_[np.arange(-26, 0), np.arange(1, 27)]
TARGET_SUB = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                       1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])
try:
    WAVELET = pywt.Wavelet('dB11')
except ValueError:
    WAVELET = pywt.Wavelet('db11')


def load_board(board_dir, tx_id=None):
    """Return (mac_t sorted float64 (R,), amplitude (R, 30) float32) for all uploads of one board.

    tx_id selects one injector (record flags bits 4..7; format-1 records, id 0, count as injector 1).
    """
    times, amps, keys = [], [], []
    files = sorted(Path(board_dir).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'no .npz uploads in {board_dir}')
    for f in files:
        with np.load(f) as z:
            rec = z['records']
            if tx_id is not None:
                ids = rec['flags'] >> 4
                sel = np.where(ids == 0, 1, ids) == tx_id
                rec = rec[sel]
            if len(rec) == 0:
                continue
            if 'mac_t' in z:  # live-stream chunk (hub.py): Mac time per record, chunks never overlap
                mac_t = z['mac_t'] if tx_id is None else z['mac_t'][sel]
                times.append(mac_t)
                keys.append(np.round(mac_t * 1e6).astype(np.int64))
            else:
                dev_now = int(z['dev_now_us'])
                hdr = float(z['header_mac_time'])
                # unwrap the 32-bit board time relative to dev_now (records are at or before it)
                delta = (np.int64(dev_now & 0xFFFFFFFF) - rec['t_us'].astype(np.int64)) % (1 << 32)
                times.append(hdr - delta / 1e6)
                keys.append(dev_now - delta)  # full board time: identical for a record across uploads
            iq = rec['csi'].astype(np.float32).reshape(-1, 52, 2)
            amp = np.hypot(iq[..., 1], iq[..., 0])  # |real + j*imag|, order +1..+26, -26..-1
            amp = np.concatenate([amp[:, 26:], amp[:, :26]], axis=1)  # -> -26..-1, +1..+26
            amps.append(np.stack([np.interp(TARGET_SUB, SUB_ORDER, a) for a in amp]))
    if not times:
        raise ValueError(f'no records in {board_dir}' + (f' from tx {tx_id}' if tx_id else ''))
    t = np.concatenate(times)
    k = np.concatenate(keys)
    a = np.concatenate(amps).astype(np.float32)
    _, first = np.unique(k, return_index=True)  # drop records repeated by overlapping uploads
    t, a = t[first], a[first]
    order = np.argsort(t, kind='stable')
    return t[order], a[order]


def smooth_window(win):
    """win: (20, 30) time-major amplitude -> wavelet-smoothed, mean-normalised window."""
    rec = pywt.waverec(pywt.wavedec(win, WAVELET, 'sym', axis=0), WAVELET, axis=0)[:WIN]
    return (rec / (rec.mean() + 1e-6)).astype(np.float32)


def convert_session(name, csi_root, labels_dir, boards, keep_empty, tx_ids=None, max_age=MAX_AGE_S):
    links = [(b, t) for b in boards for t in (tx_ids or [None])]
    boards_data = [load_board(Path(csi_root) / b, t) for b, t in links]
    n_tx = len(tx_ids or [None])
    with np.load(Path(labels_dir) / 'labels.npz') as z:
        t_mac = z['t_mac'].astype(np.float64)
        kpts = z['keypoints'].astype(np.float32)
        nump = z['num_person'].astype(np.int64)
        kpts3d = z['keypoints3d'].astype(np.float32) if 'keypoints3d' in z else None
    skipped = {'no_csi_window': 0, 'no_person': 0}
    csi, kp, npers, names, kp3 = [], [], [], [], []
    for i, T in enumerate(t_mac):
        if nump[i] == 0 and not keep_empty:
            skipped['no_person'] += 1
            continue
        wins = []
        for bt, ba in boards_data:
            end = np.searchsorted(bt, T, side='right')  # records with mac_t <= T
            if end < WIN or bt[end - WIN] < T - max_age:
                wins = None
                break
            wins.append(smooth_window(ba[end - WIN:end]))
        if wins is None:
            skipped['no_csi_window'] += 1
            continue
        csi.append(np.stack(wins).reshape(len(boards), n_tx, WIN, 30))
        kp.append(kpts[i, :, :, :2])
        if kpts3d is not None:
            kp3.append(kpts3d[i, :, :, :3])
        npers.append(nump[i])
        names.append(f'{name}_{i}')
    info = dict(frames=len(t_mac), kept=len(csi), skipped=skipped,
                time_span=[float(t_mac.min()), float(t_mac.max())] if len(t_mac) else None)
    return csi, kp, npers, names, info, kp3


def write_cache(out, boards, parts, sessions_info, kp3_parts=(), n_tx=1):
    out.mkdir(parents=True, exist_ok=True)
    p_max = max([k.shape[0] for _, kp, _, _ in parts for k in kp] or [1])
    csi = np.stack([c for p in parts for c in p[0]]).astype(np.float32) if any(p[0] for p in parts) \
        else np.zeros((0, len(boards), n_tx, WIN, 30), np.float32)
    keypoints = np.zeros((len(csi), p_max, N_KPT, 2), np.float32)
    n = 0
    for _, kp, _, _ in parts:
        for k in kp:
            keypoints[n, :k.shape[0]] = k
            n += 1
    num_person = np.array([x for p in parts for x in p[2]], dtype=np.int16)
    names = np.array([x for p in parts for x in p[3]], dtype=str)
    np.save(out / 'csi.npy', csi)
    np.save(out / 'keypoints.npy', keypoints)
    np.save(out / 'num_person.npy', num_person)
    np.save(out / 'names.npy', names)
    kp3 = [k for p in kp3_parts for k in p]
    if kp3 and len(kp3) == len(csi):
        k3 = np.zeros((len(csi), p_max, N_KPT, 3), np.float32)
        for i, k in enumerate(kp3):
            k3[i, :k.shape[0]] = k
        np.save(out / 'keypoints3d.npy', k3)
    tot_skip = {}
    for s in sessions_info.values():
        for k, v in s['skipped'].items():
            tot_skip[k] = tot_skip.get(k, 0) + v
    spans = [s['time_span'] for s in sessions_info.values() if s['time_span']]
    meta = dict(boards=boards, counts=dict(frames_kept=int(len(csi)), p_max=int(p_max),
                                          frames_total=sum(s['frames'] for s in sessions_info.values())),
                skipped=tot_skip, sessions=sessions_info,
                time_span=[min(s[0] for s in spans), max(s[1] for s in spans)] if spans else None,
                csi_shape=list(csi.shape))
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boards', required=True, help='comma separated board names, fixed receiver order')
    ap.add_argument('--session', action='append', required=True, metavar='CSI_ROOT:LABELS_DIR')
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-empty', action='store_true', help='keep frames with num_person == 0')
    ap.add_argument('--split-by-session', default='', help='comma separated session names for the test split')
    ap.add_argument('--tx-ids', default='', help='comma separated injector ids, e.g. 1,2,3 -> csi (N, boards, tx, 20, 30)')
    ap.add_argument('--max-age', type=float, default=None,
                    help='oldest record allowed in a window, s (default 0.5; 0.7 with --tx-ids, since the weakest '
                         'links deliver only ~35-50 of each transmitter\'s 100 packets/s)')
    args = ap.parse_args()
    tx_ids = [int(t) for t in args.tx_ids.split(',') if t] or None
    n_tx = len(tx_ids or [None])
    max_age = args.max_age or (0.7 if tx_ids else MAX_AGE_S)

    boards = [b for b in args.boards.split(',') if b]
    test_names = {s for s in args.split_by_session.split(',') if s}
    results = {}
    for spec in args.session:
        csi_root, labels_dir = spec.rsplit(':', 1)
        labels_path = Path(labels_dir.rstrip('/'))
        name = labels_path.parent.name if labels_path.name == 'labels' else labels_path.name
        if name in results:
            raise SystemExit(f'duplicate session name {name!r} (taken from the labels directory name)')
        res = convert_session(name, csi_root, labels_dir, boards, args.keep_empty, tx_ids, max_age)
        results[name] = res
        print(f'[{name}] kept {res[4]["kept"]}/{res[4]["frames"]} frames, skipped {res[4]["skipped"]}')
    unknown = test_names - set(results)
    if unknown:
        raise SystemExit(f'--split-by-session names not among sessions: {sorted(unknown)}')

    out = Path(args.out)
    if test_names:
        for split, pick in (('train', lambda n: n not in test_names), ('test', lambda n: n in test_names)):
            names = [n for n in results if pick(n)]
            meta = write_cache(out / split, boards, [results[n][:4] for n in names],
                               {n: results[n][4] for n in names}, [results[n][5] for n in names], n_tx)
            print(f'{split}: {meta["counts"]["frames_kept"]} frames from {names}')
    else:
        meta = write_cache(out, boards, [r[:4] for r in results.values()], {n: r[4] for n, r in results.items()},
                           [r[5] for r in results.values()], n_tx)
        print(f'wrote {meta["counts"]["frames_kept"]} frames to {out}')


if __name__ == '__main__':
    main()
