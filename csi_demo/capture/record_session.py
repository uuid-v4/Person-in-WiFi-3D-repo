#!/usr/bin/env python3
"""Record one synced dataset session: webcam 2D+3D pose (MediaPipe) and BLE CSI from every receiver.

    python csi_demo/capture/record_session.py --name s1 --minutes 10 [--chunk 15]
Live view: http://localhost:8766/data   (stop early: Ctrl-C or the Stop button)

Output data/sessions/<name>/:
  csi/<board>/*.npz      collector-format uploads (board clock + Mac arrival time per upload)
  labels/labels.npz      t_mac (F,), keypoints (F,1,14,3) [x_norm, y_norm, visibility],
                         keypoints3d (F,1,14,4) [x, y, z metres, hip-centred camera axes; visibility],
                         num_person (F,), image_size [w, h]  (same layout as pose_labeler.py + 3D)
  video.mp4, meta.json
Boards are cleared after every upload, so each chunk holds only new records; records captured
while a board is uploading are lost (gaps of a few seconds per chunk, all boards at once).
"""
import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from bleak import BleakScanner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'csi_demo' / 'ble'))
from collector import CTRL_UUID, Board  # noqa: E402

HERE = Path(__file__).parent
BOARDS = ['CSI-298988', 'CSI-wroom1', 'CSI-wroom2']
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/'
             'float16/latest/pose_landmarker_full.task')
MODEL_PATH = HERE / 'pose_landmarker_full.task'
# MediaPipe 33-landmark index for each of the paper's 14 keypoints; -1 = neck (shoulder midpoint).
MP_FOR_SLOT = [-1, 0, 11, 12, 13, 23, 14, 24, 15, 25, 16, 26, 27, 28]
PI_SSH = ['ssh', '-i', str(Path.home() / '.ssh/id_csi_pi'), '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
          'sijin@192.168.0.181']

LIVE = {'started': None, 'kp2d': None, 'kp3d': None, 'fps': 0.0, 'frames': 0, 'posed': 0,
        'csi': {}, 'chunks': 0, 'status': 'starting', 'pi': '', 'stop': False, 'end': None}
JPEG = {'buf': b''}


def to14(lms, with_xyz):
    out = np.zeros((14, 4 if with_xyz else 3), np.float32)
    for slot, i in enumerate(MP_FOR_SLOT):
        pts = [lms[11], lms[12]] if i < 0 else [lms[i]]
        out[slot, 0] = np.mean([p.x for p in pts])
        out[slot, 1] = np.mean([p.y for p in pts])
        if with_xyz:
            out[slot, 2] = np.mean([p.z for p in pts])
        out[slot, -1] = min(p.visibility for p in pts)
    return out


def camera_loop(args, out, labels):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        LIVE['status'] = f'cannot open camera {args.camera} (grant camera access to the terminal app)'
        LIVE['stop'] = True
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    labels['image_size'] = [w, h]
    writer = cv2.VideoWriter(str(out / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
    t_fps = time.time()
    n_fps = 0
    while not LIVE['stop']:
        ok, frame = cap.read()
        t = time.time()
        if not ok:
            continue
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(t * 1000))
        k2 = np.zeros((14, 3), np.float32)
        k3 = np.zeros((14, 4), np.float32)
        posed = bool(res.pose_landmarks)
        if posed:
            k2 = to14(res.pose_landmarks[0], False)
            k2[:, :2] = np.clip(k2[:, :2], 0.0, 1.0)
            k3 = to14(res.pose_world_landmarks[0], True)
            k3[:, 3] = k2[:, 2]
        labels['t'].append(t)
        labels['k2'].append(k2)
        labels['k3'].append(k3)
        labels['n'].append(int(posed))
        writer.write(frame)
        LIVE['frames'] += 1
        LIVE['posed'] += int(posed)
        LIVE['kp2d'] = np.round(k2, 4).tolist() if posed else None
        LIVE['kp3d'] = np.round(k3, 3).tolist() if posed else None
        JPEG['buf'] = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
        n_fps += 1
        if t - t_fps > 2:
            LIVE['fps'] = round(n_fps / (t - t_fps), 1)
            t_fps, n_fps = t, 0
    cap.release()
    writer.release()


def chunk_view(path, max_rows=300):
    z = np.load(path)
    recs = z['records']
    if len(recs) == 0:
        return {'n': 0, 'amp': [], 'rssi': None, 'hz': 0}
    amp = np.hypot(*recs['csi'].astype(np.float32).reshape(len(recs), 52, 2).transpose(2, 0, 1))
    span = (int(recs['t_us'][-1]) - int(recs['t_us'][0])) / 1e6
    step = max(1, len(recs) // max_rows)
    return {'n': int(len(recs)), 'amp': np.round(amp[::step], 1).tolist(), 'rssi': int(np.median(recs['rssi'])),
            'hz': round(len(recs) / max(span, 1e-3), 1)}


async def upload(b):
    path = await b.dump(timeout=300)
    await b.client.write_gatt_char(CTRL_UUID, b'CLEAR', response=True)
    LIVE['csi'][b.name] = dict(chunk_view(path), total=LIVE['csi'].get(b.name, {}).get('total', 0)
                               + int(len(np.load(path)['records'])))


async def ble_loop(args, csi_dir):
    found = await BleakScanner.discover(timeout=8, return_adv=True)
    devs = {(adv.local_name or d.name or ''): d for d, adv in found.values()}
    missing = [b for b in BOARDS if b not in devs]
    if missing:
        LIVE['status'] = f'boards not found: {missing}'
        LIVE['stop'] = True
        return
    boards = [Board(devs[n], n, csi_dir) for n in BOARDS]
    for b in boards:
        await b.connect()
        await b.client.write_gatt_char(CTRL_UUID, b'CLEAR', response=True)
    LIVE['status'] = 'recording'
    while True:
        deadline = time.time() + args.chunk
        while time.time() < deadline and not LIVE['stop'] and time.time() < LIVE['end']:
            await asyncio.sleep(0.5)
        LIVE['status'] = 'uploading CSI'
        results = await asyncio.gather(*(upload(b) for b in boards), return_exceptions=True)
        for b, r in zip(boards, results):
            if isinstance(r, Exception):
                print(f'[{b.name}] upload failed: {r!r}')
        LIVE['chunks'] += 1
        if LIVE['stop'] or time.time() >= LIVE['end']:
            break
        LIVE['status'] = 'recording'
    LIVE['stop'] = True
    for b in boards:
        await b.client.disconnect()


def pi_loop():
    while not LIVE['stop']:
        r = subprocess.run(PI_SSH + ['systemctl is-active csi-tx; journalctl -u csi-tx -n 1 -o cat --no-pager'],
                           capture_output=True, text=True)
        LIVE['pi'] = ' | '.join((r.stdout or r.stderr).split('\n')).strip(' |')
        time.sleep(10)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/data/video'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
            self.end_headers()
            try:
                while True:
                    buf = JPEG['buf']
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                     + str(len(buf)).encode() + b'\r\n\r\n' + buf + b'\r\n')
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path.startswith('/data/state'):
            body, ctype = json.dumps(dict(LIVE, now=time.time())).encode(), 'application/json'
        elif self.path.startswith('/data'):
            body, ctype = (HERE / 'data.html').read_bytes(), 'text/html'
        else:
            self.send_response(302)
            self.send_header('Location', '/data')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith('/data/stop'):
            LIVE['stop'] = True
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--minutes', type=float, default=10)
    ap.add_argument('--chunk', type=float, default=15, help='seconds of CSI per BLE upload')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--root', default=str(ROOT / 'data' / 'sessions'))
    ap.add_argument('--port', type=int, default=8766)
    args = ap.parse_args()

    out = Path(args.root) / args.name
    if out.exists():
        raise SystemExit(f'{out} exists; pick another --name')
    (out / 'labels').mkdir(parents=True)
    labels = {'t': [], 'k2': [], 'k3': [], 'n': [], 'image_size': [0, 0]}
    LIVE['started'] = time.time()
    LIVE['end'] = LIVE['started'] + args.minutes * 60

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cam = threading.Thread(target=camera_loop, args=(args, out, labels), daemon=True)
    cam.start()
    threading.Thread(target=pi_loop, daemon=True).start()
    print(f'live view: http://localhost:{args.port}/data  -> {out}')
    try:
        asyncio.run(ble_loop(args, out / 'csi'))
    except KeyboardInterrupt:
        LIVE['stop'] = True
    LIVE['stop'] = True
    cam.join(timeout=10)

    n = len(labels['t'])
    np.savez(out / 'labels' / 'labels.npz', t_mac=np.array(labels['t'], np.float64),
             keypoints=np.array(labels['k2'], np.float32).reshape(n, 1, 14, 3),
             keypoints3d=np.array(labels['k3'], np.float32).reshape(n, 1, 14, 4),
             num_person=np.array(labels['n'], np.int16), image_size=np.array(labels['image_size'], np.int32))
    meta = dict(name=args.name, created=datetime.now().isoformat(timespec='seconds'), boards=BOARDS,
                frames=n, frames_with_person=int(sum(labels['n'])), chunks=LIVE['chunks'],
                csi_records={k: v.get('total', 0) for k, v in LIVE['csi'].items()},
                pose_model='mediapipe pose_landmarker_full', status=LIVE['status'],
                layout='rectangle 2 m (left/right) x 1 m (top/bottom), chest height; Mac top-middle; '
                       'n16r8 top-left, Pi tx top-right, wroom2 bottom-left, wroom1 bottom-right')
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
