#!/usr/bin/env python3
"""Live CSI dashboard: reads ESP32 csi_recv serial output and streams it to a browser.

    pip install pyserial
    python3 server.py                      # auto-detects /dev/cu.usbmodem* and /dev/ttyACM*
    python3 server.py --ports /dev/cu.usbmodem101 /dev/cu.usbmodem1101
Then open http://localhost:8765
"""
import argparse
import glob
import json
import math
import pathlib
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import serial

HERE = pathlib.Path(__file__).parent
TX_MAC = "02:c5:1a:00:00:01"
# ESP32 LLTF layout: buffer index k holds subcarrier k (0..31) or k-64 (32..63).
DATA_SUBCARRIERS = list(range(-26, 0)) + list(range(1, 27))
BUFFER_INDEX = [sc if sc > 0 else sc + 64 for sc in DATA_SUBCARRIERS]
MOTION_WINDOW_S = 1.0


class Receiver:
    def __init__(self, rx_id, port):
        self.id = rx_id
        self.port = port
        self.lock = threading.Lock()
        self.pending = []
        self.history = deque(maxlen=400)
        self.arrivals = deque()
        self.rssi = None
        self.rate_code = None
        self.sig_mode = None
        self.status = "connecting"
        self.mac_seen = None

    def run(self):
        while True:
            try:
                ser = serial.Serial()
                ser.port, ser.baudrate, ser.timeout = self.port, 115200, 1
                ser.dtr = ser.rts = False
                ser.open()
                self.status = "live"
                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line.startswith("CSI_DATA"):
                        self._ingest(line)
            except (serial.SerialException, OSError) as exc:
                self.status = f"disconnected: {exc.__class__.__name__}"
                time.sleep(2)

    def _ingest(self, line):
        head, _, data = line.partition(',"[')
        fields = head.split(",")
        try:
            values = [int(v) for v in data.rstrip(']"').split(",")]
        except ValueError:
            return
        if len(values) < 128 or len(fields) < 6:
            return
        amps = []
        for k in BUFFER_INDEX:
            imag, real = values[2 * k], values[2 * k + 1]
            amps.append(round(math.hypot(real, imag), 2))
        now = time.time()
        with self.lock:
            self.mac_seen = fields[2]
            self.rssi = int(fields[3])
            self.rate_code = int(fields[4])
            self.sig_mode = int(fields[5])
            self.pending.append(amps)
            self.history.append((now, amps))
            self.arrivals.append(now)
            while self.arrivals and now - self.arrivals[0] > 1.0:
                self.arrivals.popleft()

    def snapshot(self):
        now = time.time()
        with self.lock:
            frames, self.pending = self.pending, []
            recent = [a for t, a in self.history if now - t <= MOTION_WINDOW_S]
            while self.arrivals and now - self.arrivals[0] > 1.0:
                self.arrivals.popleft()
            rate_hz = len(self.arrivals)
            rssi, rate_code, sig_mode, status = self.rssi, self.rate_code, self.sig_mode, self.status
        return {
            "id": self.id,
            "port": self.port,
            "status": status,
            "frames": frames[-20:],
            "rate_hz": rate_hz,
            "rssi": rssi,
            "rate_code": rate_code,
            "sig_mode": sig_mode,
            "motion": motion_score(recent),
        }


def motion_score(frames):
    """Mean over subcarriers of the coefficient of variation of amplitude within the window."""
    if len(frames) < 10:
        return None
    n = len(frames)
    total = 0.0
    for sc in range(len(frames[0])):
        col = [f[sc] for f in frames]
        mean = sum(col) / n
        if mean <= 0:
            continue
        var = sum((x - mean) ** 2 for x in col) / n
        total += math.sqrt(var) / mean
    return round(total / len(frames[0]), 4)


def make_handler(receivers):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        payload = {
                            "t": time.time(),
                            "tx_mac": TX_MAC,
                            "subcarriers": DATA_SUBCARRIERS,
                            "receivers": [r.snapshot() for r in receivers],
                        }
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_error(404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="*")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    ports = args.ports or sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        raise SystemExit("no ESP32 serial ports found")
    receivers = [Receiver(f"S3-{i + 1}", p) for i, p in enumerate(ports)]
    for r in receivers:
        threading.Thread(target=r.run, daemon=True).start()

    print(f"receivers: {', '.join(f'{r.id}={r.port}' for r in receivers)}")
    print(f"dashboard: http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), make_handler(receivers)).serve_forever()


if __name__ == "__main__":
    main()
