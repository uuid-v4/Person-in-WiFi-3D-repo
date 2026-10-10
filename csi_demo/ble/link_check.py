#!/usr/bin/env python3
"""Per-link CSI validity check: does every (receiver, transmitter) link deliver OFDM CSI that reacts to a person?

    python csi_demo/ble/link_check.py [--secs 30]

Phase 1 "empty": nobody inside the rectangle. Phase 2 "moving": walk / wave inside it.
For each link it prints packets/s, RSSI, the OFDM rate code (11 = 6 Mbps legacy), and the motion
ratio = mean over subcarriers of the amplitude std while moving / std while empty. A link that
senses the person shows a ratio clearly above 1 (typically 2-10); ~1 means it does not see you.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from bleak import BleakScanner

sys.path.insert(0, str(Path(__file__).parent))
from collector import CTRL_UUID, Board  # noqa: E402

BOARDS = ['CSI-298988', 'CSI-wroom1', 'CSI-wroom2']


async def capture(boards, secs, label):
    for b in boards:
        await b.client.write_gatt_char(CTRL_UUID, b'CLEAR', response=True)
    print(f'--- {label}: recording {secs:.0f} s', flush=True)
    await asyncio.sleep(secs)
    out = {}
    for b in boards:
        out[b.name] = np.load(await b.dump())['records']
    return out


def links(recs):
    ids = recs['flags'] >> 4
    for t in sorted(set(np.where(ids == 0, 1, ids).tolist())):
        r = recs[np.where(ids == 0, 1, ids) == t]
        amp = np.hypot(*r['csi'].astype(np.float32).reshape(len(r), 52, 2).transpose(2, 0, 1))
        yield t, r, amp


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=30)
    ap.add_argument('--gap', type=float, default=15, help='seconds to walk in between phases')
    args = ap.parse_args()
    found = await BleakScanner.discover(timeout=8, return_adv=True)
    devs = {(a.local_name or d.name or ''): d for d, a in found.values()}
    boards = [Board(devs[n], n, Path('/tmp/link_check')) for n in BOARDS if n in devs]
    print('boards:', [b.name for b in boards], 'missing:', [n for n in BOARDS if n not in devs])
    for b in boards:
        await b.connect()
    print('Stay OUT of the rectangle', flush=True)
    empty = await capture(boards, args.secs, 'empty')
    print(f'Step IN now and keep walking / waving (recording starts in {args.gap:.0f} s)', flush=True)
    await asyncio.sleep(args.gap)
    moving = await capture(boards, args.secs, 'moving')
    print(f'\n{"link":22s} {"pkt/s":>6s} {"rssi":>5s} {"rate":>5s} {"motion ratio":>12s}')
    for b in boards:
        e = {t: (r, a) for t, r, a in links(empty[b.name])}
        for t, r, a in links(moving[b.name]):
            ratio = float(a.std(0).mean() / e[t][1].std(0).mean()) if t in e and len(e[t][1]) > 20 else float('nan')
            print(f'{b.name + " <- tx" + str(t):22s} {len(r) / args.secs:6.1f} {int(np.median(r["rssi"])):5d} '
                  f'{int(np.median(r["rate"])):5d} {ratio:12.2f}')
    for b in boards:
        await b.client.disconnect()

asyncio.run(main())
