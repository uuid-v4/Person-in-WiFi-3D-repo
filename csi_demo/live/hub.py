#!/usr/bin/env python3
"""Mac-side live CSI hub: ESP32s on local USB + records forwarded by Pis over UDP -> per-link live view.

    csi_demo/.venv/bin/python csi_demo/live/hub.py [--camera 1]   # then open http://localhost:8766/live

Also runs the webcam pose model (MediaPipe 2D+3D) continuously and records sessions on demand
(page buttons or POST /live/start?name=s1&minutes=10, /live/stop) into data/sessions/<name>/:
  csi/<board>/<n>.npz   records (collector dtype) + mac_t (Mac-clock time per record), 10 s chunks
  labels/labels.npz     t_mac, keypoints (F,1,14,3), keypoints3d (F,1,14,4), num_person, image_size
  video.mp4, meta.json

A link is (receiver board, transmitter id). Board time (t_us) is mapped to the Mac clock per board with
the minimum observed (arrival - board) offset over the last 10 s, which removes USB/UDP delay jitter.
"""
import argparse
import glob
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / 'capture'))
from csi_frames import REC_DTYPE, FrameParser  # noqa: E402
import windowing  # noqa: E402
from record_session import MODEL_PATH, MODEL_URL, to14  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
LAYOUT = ('2 m (left/right) x 1 m (top/bottom) rectangle, chest height; Mac + camera top-middle; '
          'rx: n16r8 top-left (Mac USB), wroom1 bottom-right, wroom2 bottom-left (rpi2 USB); '
          'tx1 PAU0F top-right (rpi1), tx2/tx3 RTL8192EU (rpi2)')
KEEP_S = 6.0
PIS = {'rpi1 (PAU0F tx1)': '192.168.0.181', 'rpi2 (RTL tx2, tx3)': '192.168.0.113'}
SSH = ['ssh', '-i', str(Path.home() / '.ssh/id_csi_pi'), '-o', 'ConnectTimeout=4', '-o', 'BatchMode=yes']


class Clock:
    def __init__(self):
        self.wraps, self.last, self.samples = 0, None, deque()

    def to_mac(self, t_us, arrival):
        if self.last is not None and t_us < self.last and self.last - t_us > 2 ** 31:
            self.wraps += 1
        self.last = t_us
        board = (t_us + self.wraps * 2 ** 32) / 1e6
        self.samples.append((arrival, arrival - board))
        while self.samples and arrival - self.samples[0][0] > 10:
            self.samples.popleft()
        return board + min(o for _, o in self.samples)


LINKS = defaultdict(lambda: deque())
CLOCKS = defaultdict(Clock)
SOURCES = {}
PI_STATUS = {}
LOCK = threading.Lock()
CAM = {'kp2d': None, 'kp3d': None, 'fps': 0.0, 'jpeg': b'', 'error': None}
REC = {'active': False}
REC_LOCK = threading.Lock()


def add(name, rec_bytes, arrival, source):
    r = np.frombuffer(rec_bytes, REC_DTYPE)[0]
    tx = int(r['flags']) >> 4 or 1
    iq = r['csi'].astype(np.float32).reshape(52, 2)
    amp = np.hypot(iq[:, 0], iq[:, 1])
    amp = np.concatenate([amp[26:], amp[:26]])
    with LOCK:
        t = CLOCKS[name].to_mac(int(r['t_us']), arrival)
        q = LINKS[(name, tx)]
        q.append((t, amp, int(r['rssi']), int(r['seq']), int(r['rate']), r['csi'].copy()))
        while q and t - q[0][0] > KEEP_S:
            q.popleft()
        SOURCES[name] = source
    if REC['active']:
        with REC_LOCK:
            if REC['active']:
                REC['csi'].setdefault(name, []).append((rec_bytes, t))
                REC['counts'][f'{name} tx{tx}'] = REC['counts'].get(f'{name} tx{tx}', 0) + 1


def flush_csi():
    with REC_LOCK:
        pending, REC['csi'] = REC['csi'], {}
        out = REC['dir'] / 'csi'
        REC['chunk'] += 1
        n = REC['chunk']
    for name, items in pending.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        recs = np.frombuffer(b''.join(r for r, _ in items), REC_DTYPE)
        np.savez(d / f'{n:05d}.npz', records=recs, mac_t=np.array([t for _, t in items], np.float64))


def start_session(name, minutes):
    out = ROOT / 'data' / 'sessions' / name
    if REC['active']:
        return f"cannot start {name!r}: already recording {REC['name']!r}"
    if not re.fullmatch(r'[\w-]+', name):
        return f'cannot start {name!r}: use letters, digits, _ or -'
    if out.exists():
        return f'cannot start {name!r}: data/sessions/{name} already exists, pick another name'
    (out / 'labels').mkdir(parents=True)
    with REC_LOCK:
        REC.update(active=True, name=name, dir=out, start=time.time(), end=time.time() + minutes * 60,
                   csi={}, counts={}, chunk=0, t=[], k2=[], k3=[], n=[], frames=0, writer=None, size=None)
    return f'recording {name} for {minutes} min'


def stop_session():
    if not REC['active']:
        return 'not recording'
    with REC_LOCK:
        REC['active'] = False
        writer = REC.pop('writer')
    if writer:
        writer.release()
    flush_csi()
    n = len(REC['t'])
    np.savez(REC['dir'] / 'labels' / 'labels.npz', t_mac=np.array(REC['t'], np.float64),
             keypoints=np.array(REC['k2'], np.float32).reshape(n, 1, 14, 3),
             keypoints3d=np.array(REC['k3'], np.float32).reshape(n, 1, 14, 4),
             num_person=np.array(REC['n'], np.int16), image_size=np.array(REC['size'] or (0, 0), np.int32))
    meta = dict(name=REC['name'], created=datetime.now().isoformat(timespec='seconds'), source='live USB stream',
                seconds=round(time.time() - REC['start'], 1), frames=n, frames_with_person=int(sum(REC['n'])),
                csi_records=REC['counts'], tx_ids=[1, 2, 3], layout=LAYOUT, pose_model='mediapipe pose_landmarker_full')
    (REC['dir'] / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('saved', json.dumps(meta), flush=True)
    return f"saved {REC['name']}: {n} frames"


def recorder_loop():
    while True:
        time.sleep(10)
        if REC['active']:
            flush_csi()
            if time.time() >= REC['end']:
                stop_session()


def camera_loop(index):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)), running_mode=vision.RunningMode.VIDEO,
        num_poses=1))
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        CAM['error'] = f'cannot open camera {index}'
        return
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    t_fps, n_fps = time.time(), 0
    while True:
        ok, frame = cap.read()
        t = time.time()
        if not ok:
            continue
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(t * 1000))
        posed = bool(res.pose_landmarks)
        k2, k3 = np.zeros((14, 3), np.float32), np.zeros((14, 4), np.float32)
        if posed:
            k2 = to14(res.pose_landmarks[0], False)
            k2[:, :2] = np.clip(k2[:, :2], 0.0, 1.0)
            k3 = to14(res.pose_world_landmarks[0], True)
            k3[:, 3] = k2[:, 2]
        CAM['kp2d'] = np.round(k2, 4).tolist() if posed else None
        CAM['kp3d'] = np.round(k3, 3).tolist() if posed else None
        CAM['jpeg'] = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
        if REC['active']:
            with REC_LOCK:
                if REC['active']:
                    if REC['writer'] is None:
                        REC['writer'] = cv2.VideoWriter(str(REC['dir'] / 'video.mp4'),
                                                        cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
                        REC['size'] = size
                    REC['writer'].write(frame)
                    REC['t'].append(t)
                    REC['k2'].append(k2)
                    REC['k3'].append(k3)
                    REC['n'].append(int(posed))
        n_fps += 1
        if t - t_fps > 2:
            CAM['fps'] = round(n_fps / (t - t_fps), 1)
            t_fps, n_fps = t, 0


def serial_loop(path):
    while True:
        try:
            ser = serial.Serial(path, 115200, timeout=0.2)
            parser, name, text = FrameParser(), None, b''
            while True:
                data = ser.read(4096)
                now = time.time()
                if name is None:
                    text = (text + data)[-4000:]
                    m = re.search(rb'name=(CSI-[\w-]+)', text)
                    if m:
                        name = m.group(1).decode()
                        print(f'{path}: {name}', flush=True)
                for rec in parser.feed(data):
                    if name:
                        add(name, rec, now, 'Mac USB')
        except (OSError, serial.SerialException) as exc:
            print(f'{path}: {exc}', flush=True)
            time.sleep(2)


def serial_watch():
    seen = set()
    while True:
        for p in glob.glob('/dev/cu.usbmodem*'):
            if p not in seen:
                seen.add(p)
                threading.Thread(target=serial_loop, args=(p,), daemon=True).start()
        time.sleep(3)


def udp_loop(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', port))
    while True:
        d, addr = s.recvfrom(2048)
        name = d[:16].rstrip(b'\0').decode()
        add(name, d[24:], time.time(), f'Pi {addr[0]} USB')


def pi_loop():
    while True:
        for label, ip in PIS.items():
            r = subprocess.run(SSH + [f'sijin@{ip}', 'systemctl is-active csi-tx; journalctl -u csi-tx -n 3 -o cat --no-pager'],
                               capture_output=True, text=True)
            PI_STATUS[label] = ' | '.join((r.stdout or r.stderr).strip().splitlines())
        time.sleep(10)


def snapshot(rows=80):
    now = time.time()
    links = []
    with LOCK:
        items = [(k, list(q)) for k, q in LINKS.items()]
    for (name, tx), q in sorted(items):
        recent = [x for x in q if now - x[0] < 2.0]
        hz = len(recent) / 2.0
        amp = np.array([x[1] for x in q]) if q else np.zeros((0, 52))
        norm = amp / (amp.mean(1, keepdims=True) + 1e-6) if len(amp) else amp
        step = max(1, len(norm) // rows)
        slow = 0.0
        if len(norm) > 30:
            k = np.ones(10) / 10
            slow = float(np.mean([np.convolve(norm[-200:, s], k, 'valid').std() for s in range(52)]))
        links.append(dict(rx=name, tx=tx, hz=round(hz, 1), age=round(now - q[-1][0], 2) if q else None,
                          rssi=int(np.median([x[2] for x in recent])) if recent else None,
                          rate=q[-1][4] if q else None, seq=q[-1][3] if q else None,
                          source=SOURCES.get(name, ''), motion=round(slow, 4),
                          amp=np.round(norm[::step], 2).tolist(), cur=np.round(amp[-1], 1).tolist() if len(amp) else []))
    rec = None
    if REC['active']:
        rec = dict(name=REC['name'], elapsed=round(now - REC['start']), left=round(REC['end'] - now),
                   frames=len(REC['t']), posed=int(sum(REC['n'])), counts=REC['counts'])
    return dict(now=now, links=links, pis=PI_STATUS, rec=rec, last=REC.get('last'), wifi=WIFI,
                cam={k: v for k, v in CAM.items() if k != 'jpeg'})


WINDOW_CFG = {'boards': ['CSI-298988', 'CSI-wroom1', 'CSI-wroom2'], 'tx_ids': [1, 2, 3], 'max_age': 0.7}
WIFI = {'status': 'off'}


def infer_loop(args):
    sys.path.insert(0, str(HERE.parent / 'infer'))
    from runtime import PoseRuntime
    rt = None
    for backend in ([args.backend, 'onnxruntime'] if args.backend == 'clikart' else [args.backend]):
        try:
            rt = PoseRuntime(args.model, backend, args.license)
            WIFI.update(backend=backend, status='waiting for CSI')
            break
        except Exception as exc:
            WIFI['error'] = f'{backend}: {exc}'[:300]
            print(f'inference backend {backend} failed: {exc}', flush=True)
    if rt is None:
        WIFI['status'] = 'no inference backend'
        return
    smooth = None
    period = 1.0 / args.infer_hz
    while True:
        t0 = time.time()
        t, w = live_window(args.mode)
        if w is None:
            WIFI['status'] = 'waiting for CSI (every link needs 20 recent packets)'
            if time.time() - WIFI.get('last', 0) > 3:
                WIFI['kpts'], smooth = None, None
        else:
            score, kp = rt.predict(w)
            smooth = kp if smooth is None else args.smooth * smooth + (1 - args.smooth) * kp
            WIFI.update(status='live', kpts=np.round(smooth, 4).tolist(), raw=np.round(kp, 4).tolist(),
                        score=round(score, 3), t=t, latency_ms=round(1000 * (time.time() - t0), 1),
                        hz=round(1.0 / max(time.time() - WIFI.get('last', t0 - period), 1e-3), 1), last=time.time())
        time.sleep(max(0.0, period - (time.time() - t0)))


def live_window(mode):
    with LOCK:
        links = {k: (np.array([x[0] for x in q]), np.stack([x[5] for x in q]) if q else np.zeros((0, 104), np.int8))
                 for k, q in LINKS.items()}
    t = time.time()
    return t, windowing.build_window(links, WINDOW_CFG['boards'], WINDOW_CFG['tx_ids'], t, mode,
                                     WINDOW_CFG['max_age'])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if self.path.startswith('/live/start'):
            REC['last'] = start_session(q.get('name', [''])[0], float(q.get('minutes', ['10'])[0]))
        elif self.path.startswith('/live/stop'):
            REC['last'] = stop_session()
        body = json.dumps({'msg': REC.get('last')}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/live/video'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
            self.end_headers()
            try:
                while True:
                    buf = CAM['jpeg']
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(buf)).encode()
                                     + b'\r\n\r\n' + buf + b'\r\n')
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path.startswith('/live/window'):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            mode = q.get('mode', ['v1'])[0]
            t, w = live_window(mode if mode in windowing.MODES else 'v1')
            if w is None:
                body = json.dumps({'t': t, 'error': 'not enough recent packets on every link'}).encode()
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
            elif q.get('format', ['json'])[0] == 'npy':
                import io
                buf = io.BytesIO()
                np.save(buf, w)
                body = buf.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('X-Window-Time', f'{t:.3f}')
            else:
                body = json.dumps({'t': t, 'shape': list(w.shape), 'boards': WINDOW_CFG['boards'],
                                   'tx_ids': WINDOW_CFG['tx_ids'], 'csi': np.round(w, 5).tolist()}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/live/state'):
            body, ctype = json.dumps(snapshot()).encode(), 'application/json'
        elif self.path.startswith('/live'):
            body, ctype = (HERE / 'live.html').read_bytes(), 'text/html'
        else:
            self.send_response(302)
            self.send_header('Location', '/live')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--udp', type=int, default=5599)
    ap.add_argument('--camera', type=int, default=1, help='webcam index; -1 = no camera (CSI + /live/window only)')
    ap.add_argument('--model', default=str(ROOT / 'data/ckpt/petr_v2.onnx'), help='exported ONNX pose model')
    ap.add_argument('--mode', default='v2', choices=sorted(windowing.MODES), help='window preprocessing the model expects')
    ap.add_argument('--backend', default='clikart', choices=['clikart', 'onnxruntime', 'off'],
                    help='live CSI-only pose inference (clikart falls back to onnxruntime if it cannot start)')
    ap.add_argument('--license', default=str(ROOT / 'Default_Project_Wifi_Sensing.txt'), help='ClikaRT credential file')
    ap.add_argument('--infer-hz', type=float, default=4.0)
    ap.add_argument('--max-age', type=float, default=0.7,
                    help='oldest packet allowed in a live window, s (training used 0.7; raise to ~1.0 if a weak link '
                         'delivers < 30 packets/s)')
    ap.add_argument('--no-serial', action='store_true', help='ignore boards on local USB (e.g. when replaying CSI over UDP)')
    ap.add_argument('--smooth', type=float, default=0.5, help='exponential smoothing of displayed poses (0 = none)')
    ap.add_argument('--boards', default=','.join(WINDOW_CFG['boards']),
                    help='receiver order for /live/window (must match the training cache)')
    ap.add_argument('--pi', action='append', metavar='LABEL=IP',
                    help='transmitter Pi to show status for (repeatable); default: this setup\'s two Pis')
    args = ap.parse_args()
    WINDOW_CFG['boards'] = args.boards.split(',')
    WINDOW_CFG['max_age'] = args.max_age
    if args.pi:
        PIS.clear()
        PIS.update(p.split('=', 1) for p in args.pi)
    loops = [(udp_loop, (args.udp,)), (pi_loop, ()), (recorder_loop, ())]
    if not args.no_serial:
        loops.append((serial_watch, ()))
    if args.camera >= 0:
        loops.append((camera_loop, (args.camera,)))
    if args.backend != 'off':
        loops.append((infer_loop, (args,)))
    for target, a in loops:
        threading.Thread(target=target, args=a, daemon=True).start()
    print(f'live view: http://localhost:{args.port}/live', flush=True)
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
