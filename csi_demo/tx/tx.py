#!/usr/bin/env python3
"""Inject fixed-MAC broadcast data frames at a steady rate for ESP32 CSI capture.

Requires the patched mt7921 driver (csi_demo/pau0f_driver); the stock driver
ignores the radiotap rate and sends at 1 Mbps DSSS, which yields no CSI.

The patched rtl8xxxu driver (csi_demo/rtl_driver) does the same for RTL8192EU dongles.
Each injector gets its own id (MAC 02:c5:1a:00:00:0<id>) and puts a u32 sequence number
right after the "CSI-TX" marker so receivers can tell transmitters and packets apart.

Run on the transmitter Pi with the dongle in monitor mode:
    sudo python3 tx.py --iface wlan1 --rate 100 --mbps 6 [--tx-id 1]
"""
import argparse
import struct
import time

from scapy.all import RadioTap, Dot11, LLC, SNAP, Raw, conf

OFDM_MBPS = (6, 9, 12, 18, 24, 36, 48, 54)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="wlan1")
    ap.add_argument("--rate", type=float, default=100.0, help="frames per second")
    ap.add_argument("--seconds", type=float, default=0, help="0 = run forever")
    ap.add_argument("--mbps", type=int, default=6, choices=OFDM_MBPS,
                    help="legacy OFDM PHY rate; scapy's Rate field is in Mbps")
    ap.add_argument("--tx-id", type=int, default=1, choices=range(1, 16), help="injector id 1..15")
    args = ap.parse_args()
    tx_mac = f"02:c5:1a:00:00:{args.tx_id:02x}"

    frame = (
        RadioTap(present="Flags+Rate", Flags=0, Rate=args.mbps)
        / Dot11(type=2, subtype=0, addr1="ff:ff:ff:ff:ff:ff", addr2=tx_mac, addr3=tx_mac)
        / LLC() / SNAP()
        / Raw(b"CSI-TX" + b"\x00" * 58)
    )
    raw = bytearray(bytes(frame))
    seq_at = raw.index(b"CSI-TX") + 6
    sock = conf.L2socket(iface=args.iface)

    period = 1.0 / args.rate
    start = next_t = time.monotonic()
    sent = 0
    last_report = start
    while True:
        struct.pack_into("<I", raw, seq_at, sent)
        sock.send(bytes(raw))
        sent += 1
        next_t += period
        now = time.monotonic()
        if now - last_report >= 5:
            print(f"sent={sent} rate={sent / (now - start):.1f}/s", flush=True)
            last_report = now
        if args.seconds and now - start >= args.seconds:
            break
        delay = next_t - now
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = now  # fell behind; do not burst


if __name__ == "__main__":
    main()
