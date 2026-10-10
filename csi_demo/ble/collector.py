#!/usr/bin/env python3
"""Connect to every CSI-* ESP32 recorder over BLE and pull its PSRAM buffer periodically.

    python csi_demo/ble/collector.py --out recordings --interval 600
    python csi_demo/ble/collector.py --once          # one upload from every board, then exit

Each upload is saved as <out>/<board>/<UTC time>.npz with the raw records, the board's
clock at upload time, and the Mac clock when the upload header arrived (board->Mac time map).
"""
import argparse
import asyncio
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

CTRL_UUID = "c51a0002-0000-4c5a-9e00-5f1a00c5c5a1"
DATA_UUID = "c51a0003-0000-4c5a-9e00-5f1a00c5c5a1"
STAT_UUID = "c51a0004-0000-4c5a-9e00-5f1a00c5c5a1"
HDR = struct.Struct("<3sBIHQIIB")
REC_DTYPE = np.dtype([("t_us", "<u4"), ("seq", "<u4"), ("rssi", "i1"), ("rate", "u1"), ("agc", "u1"),
                      ("fft", "i1"), ("sig_mode", "u1"), ("flags", "u1"), ("csi", "i1", (104,))])


class Board:
    def __init__(self, device, name, out_dir):
        self.device = device
        self.name = name
        self.out_dir = out_dir / self.name
        self.client = None
        self.status = ""
        self._reset()

    def _reset(self):
        self.header = None
        self.header_mac_time = None
        self.records = None
        self.received = 0
        self.failed = False
        self.done = asyncio.Event()

    def _on_data(self, _, data: bytearray):
        if data[:3] == b"HDR":
            magic, ver, count, rec_size, dev_now_us, total, overwritten, channel = HDR.unpack_from(data)
            self.header = dict(version=ver, count=count, rec_size=rec_size, dev_now_us=dev_now_us,
                               total=total, overwritten=overwritten, channel=channel)
            self.header_mac_time = time.time()
            self.records = np.zeros(count, dtype=REC_DTYPE)
            if count == 0:
                self.done.set()
        elif data[:3] == b"END":
            self.done.set()
        elif data[:3] == b"ERR":
            self.failed = True
            self.done.set()
        elif self.records is not None:
            first = struct.unpack_from("<I", data)[0]
            recs = np.frombuffer(bytes(data[4:]), dtype=REC_DTYPE)
            self.records[first:first + len(recs)] = recs
            self.received += len(recs)

    def _on_status(self, _, data: bytearray):
        self.status = data.decode(errors="replace")

    async def connect(self):
        self.client = BleakClient(self.device)
        await self.client.connect()
        await self.client.start_notify(DATA_UUID, self._on_data)
        await self.client.start_notify(STAT_UUID, self._on_status)
        self.status = (await self.client.read_gatt_char(STAT_UUID)).decode(errors="replace")
        print(f"[{self.name}] connected, mtu={self.client.mtu_size}: {self.status}")

    async def dump(self, timeout=900):
        if not self.client or not self.client.is_connected:
            await self.connect()
        self._reset()
        t0 = time.time()
        await self.client.write_gatt_char(CTRL_UUID, b"DUMP", response=True)
        await asyncio.wait_for(self.done.wait(), timeout)
        if self.failed:
            raise RuntimeError(f"board aborted the upload after {self.received} records")
        dt = time.time() - t0
        h = self.header
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.out_dir / f"{stamp}.npz"
        np.savez(path, records=self.records, header_mac_time=self.header_mac_time, **h)
        recs = self.records
        span = (int(recs["t_us"][-1]) - int(recs["t_us"][0])) / 1e6 if len(recs) > 1 else 0.0
        seq_ok = float((recs["flags"] & 1).mean()) if len(recs) else 0.0
        print(f"[{self.name}] {self.received}/{h['count']} records in {dt:.1f} s "
              f"({h['count'] * h['rec_size'] / max(dt, 1e-3) / 1024:.1f} KB/s), span {span:.0f} s "
              f"(~{len(recs) / max(span, 1e-3):.0f} Hz), tx-seq in {seq_ok:.0%}, "
              f"overwritten {h['overwritten']} -> {path}")
        return path


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--interval", type=float, default=600, help="seconds between uploads")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--scan", type=float, default=8, help="seconds to scan for CSI-* boards")
    args = ap.parse_args()

    found = await BleakScanner.discover(timeout=args.scan, return_adv=True)
    named = [(dev, adv.local_name or dev.name or "") for dev, adv in found.values()]
    devices = sorted([(dev, name) for dev, name in named if name.startswith("CSI-")], key=lambda x: x[1])
    if not devices:
        raise SystemExit("no CSI-* boards found")
    boards = [Board(dev, name, Path(args.out)) for dev, name in devices]
    print("found:", ", ".join(b.name for b in boards))
    for b in boards:
        await b.connect()

    while True:
        for b in boards:
            try:
                await b.dump()
            except Exception as exc:
                print(f"[{b.name}] upload failed: {exc!r}")
        if args.once:
            break
        print(f"next upload in {args.interval:.0f} s")
        await asyncio.sleep(args.interval)

    for b in boards:
        if b.client and b.client.is_connected:
            await b.client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
