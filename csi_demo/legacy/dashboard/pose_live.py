#!/usr/bin/env python3
"""Near-live 2D Wi-Fi pose dashboard.

Loop: hold BLE connections to every CSI-* board, CLEAR their buffers, wait --record seconds,
DUMP each board, build model windows, run the 2D model on the GPU host over SSH, and show the
resulting skeletons (replayed at capture time) on http://localhost:8766.

    python csi_demo/dashboard/pose_live.py [--record 20] [--host fastgen]
"""
import argparse
import asyncio
import json
import pathlib
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from bleak import BleakScanner

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'csi_demo' / 'ble'))
from collector import CTRL_UUID, Board  # noqa: E402

HERE = pathlib.Path(__file__).parent
BOARDS = ['CSI-298988', 'CSI-wroom1', 'CSI-wroom2']
REMOTE_PY = '/root/miniconda3/envs/piw3d/bin/python'
REMOTE_DIR = '/root/piw3d'
CONFIG = 'configs/wifi/petr_wifi_esp_like_2d_150e.py'
CKPT = 'work_dirs/smoke2d/epoch_1.pth'

STATE = {'status': 'starting', 'cycle': 0, 'frames': [], 'boards': {}, 'updated': None, 'raw': {}}


def raw_amplitude(path, max_rows=400):
    recs = np.load(path)['records']
    if len(recs) == 0:
        return {'t': [], 'amp': [], 'rssi': None, 'n': 0}
    iq = recs['csi'].astype(np.float32).reshape(len(recs), 52, 2)
    amp = np.hypot(iq[..., 0], iq[..., 1])
    step = max(1, len(recs) // max_rows)
    z = np.load(path)
    mac_t = float(z['header_mac_time']) - ((int(z['dev_now_us']) - recs['t_us'].astype(np.int64)) % 2**32) / 1e6
    return {'t': np.round(mac_t[::step], 3).tolist(), 'amp': np.round(amp[::step], 1).tolist(),
            'rssi': int(np.median(recs['rssi'])), 'n': int(len(recs))}


def sh(cmd, timeout=300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{cmd[:80]}... failed ({r.returncode}): {r.stderr[-800:] or r.stdout[-800:]}')
    return r.stdout


def infer(cycle_dir, host, score):
    win = cycle_dir / 'windows.npz'
    STATE['windows_log'] = sh(f'{sys.executable} {ROOT}/csi_demo/infer/make_windows.py --csi-root {cycle_dir} '
                              f'--boards {",".join(BOARDS)} --out {win}')
    sh(f'scp -q {win} {ROOT}/csi_demo/infer/infer_2d.py {host}:/tmp/')
    sh(f'ssh {host} "cd {REMOTE_DIR} && mkdir -p csi_demo/infer && cp /tmp/infer_2d.py csi_demo/infer/ && '
       f'PYTHONPATH=. {REMOTE_PY} csi_demo/infer/infer_2d.py {CONFIG} {CKPT} /tmp/windows.npz '
       f'--every 1000000 --score {score} --out /tmp/poses.npz"')
    sh(f'scp -q {host}:/tmp/poses.npz {cycle_dir}/poses.npz')
    z = np.load(cycle_dir / 'poses.npz', allow_pickle=True)
    frames = []
    for t, s, k in zip(z['t'], z['scores'], z['keypoints']):
        keep = np.argsort(-s)[:3]
        frames.append({'t': float(t), 'people': [{'score': float(s[i]), 'kp': np.round(k[i][:, :2], 4).tolist()}
                                                 for i in keep]})
    return frames


async def loop(args):
    found = await BleakScanner.discover(timeout=8, return_adv=True)
    devs = {(adv.local_name or d.name or ''): d for d, adv in found.values()}
    missing = [b for b in BOARDS if b not in devs]
    if missing:
        raise SystemExit(f'boards not found: {missing}')
    out = pathlib.Path(args.out)
    boards = [Board(devs[n], n, out) for n in BOARDS]
    for b in boards:
        await b.connect()
    while True:
        STATE['cycle'] += 1
        cycle_dir = out / f'cycle_{STATE["cycle"]:04d}'
        for b in boards:
            b.out_dir = cycle_dir / b.name
            await b.client.write_gatt_char(CTRL_UUID, b'CLEAR', response=True)
        STATE['status'] = f'recording {args.record:.0f} s'
        await asyncio.sleep(args.record)
        for b in boards:
            STATE['status'] = f'uploading {b.name}'
            path = await b.dump(timeout=300)
            STATE['boards'][b.name] = b.status
            STATE['raw'][b.name] = raw_amplitude(path)
        STATE['status'] = 'running model'
        try:
            frames = await asyncio.to_thread(infer, cycle_dir, args.host, args.score)
            STATE.update(frames=frames, updated=time.time(), error=None)
            print(f'cycle {STATE["cycle"]}: {len(frames)} frames, latency {time.time() - frames[-1]["t"]:.0f} s')
        except Exception as exc:
            STATE['error'] = str(exc)
            print(f'cycle {STATE["cycle"]} failed: {exc}')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/state'):
            body, ctype = json.dumps(STATE).encode(), 'application/json'
        else:
            body, ctype = (HERE / 'pose.html').read_bytes(), 'text/html'
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', type=float, default=20)
    ap.add_argument('--host', default='fastgen')
    ap.add_argument('--score', type=float, default=0.3)
    ap.add_argument('--out', default='/tmp/pose_live')
    ap.add_argument('--port', type=int, default=8766)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f'dashboard: http://localhost:{args.port}')
    asyncio.run(loop(args))


if __name__ == '__main__':
    main()
