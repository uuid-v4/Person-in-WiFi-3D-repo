#!/usr/bin/env python3
"""Run on a Pi with ESP32 receivers on USB: forward their live CSI records to the Mac hub over UDP.

    python3 forwarder.py --hub 192.168.0.183 [--port 5599]

Boards are found by USB serial number (= Wi-Fi MAC). Datagram: board name (16 bytes, NUL padded)
+ float64 Pi receive time + one 118-byte record.
"""
import argparse
import glob
import os
import socket
import struct
import sys
import termios
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_frames import FrameParser  # noqa: E402

NAMES = {'30:ED:A0:A8:4E:E0': 'CSI-wroom1', '30:ED:A0:A8:4F:10': 'CSI-wroom2', 'C0:4E:30:29:89:88': 'CSI-298988'}


def open_raw(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    attrs = termios.tcgetattr(fd)
    attrs[0] = attrs[1] = attrs[3] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[6][termios.VMIN], attrs[6][termios.VTIME] = 1, 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


def pump(path, name, sock, dest, stats):
    tag = name.encode()[:16].ljust(16, b'\0')
    while True:
        try:
            fd = open_raw(path)
            parser = FrameParser()
            while True:
                data = os.read(fd, 4096)
                now = time.time()
                for rec in parser.feed(data):
                    sock.sendto(tag + struct.pack('<d', now) + rec, dest)
                    stats[name] = stats.get(name, 0) + 1
        except OSError as exc:
            print(f'{name}: {exc}; retrying', flush=True)
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hub', required=True)
    ap.add_argument('--port', type=int, default=5599)
    args = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stats = {}
    for path in sorted(glob.glob('/dev/serial/by-id/usb-Espressif_*')):
        mac = path.split('_')[-1].replace('-if00', '')
        name = NAMES.get(mac, 'CSI-' + mac.replace(':', '')[-6:])
        print(f'{name}: {path}', flush=True)
        threading.Thread(target=pump, args=(path, name, sock, (args.hub, args.port), stats), daemon=True).start()
    while True:
        time.sleep(5)
        print(' '.join(f'{k}={v / 5:.0f}/s' for k, v in sorted(stats.items())) or 'no records', flush=True)
        stats.clear()


if __name__ == '__main__':
    main()
