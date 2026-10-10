#!/usr/bin/env python3
"""Make rtl8xxxu (RTL8192EU etc.) transmit injected frames at the radiotap-requested rate.

Stock rtl8xxxu fills the 40-byte TX descriptor with rate 0 (1 Mbps CCK) and leaves data frames
to firmware rate control, so monitor-mode injection always goes out as 802.11b and ESP32s
measure no OFDM CSI from it. For frames mac80211 marks as injected, this uses the rate from the
radiotap header and forces the driver rate.

    python3 apply_patch.py drivers/net/wireless/realtek/rtl8xxxu/core.c
"""
import sys
from pathlib import Path

MARK = 'csi-demo: injected rate'
path = Path(sys.argv[1])
src = path.read_text()
if MARK in src:
    print('already patched')
    sys.exit(0)

anchor_rate = '\ttx_desc40 = (struct rtl8xxxu_txdesc40 *)tx_desc32;\n'
add_rate = anchor_rate + (
    '\n\tif (tx_info->flags & IEEE80211_TX_CTL_INJECTED) { /* ' + MARK + ' */\n'
    '\t\tstruct ieee80211_rate *txrate = ieee80211_get_tx_rate(hw, tx_info);\n\n'
    '\t\tif (txrate)\n'
    '\t\t\trate = txrate->hw_value;\n'
    '\t}\n')
anchor_force = '\tif (tx_info->flags & IEEE80211_TX_CTL_ASSIGN_SEQ)\n\t\ttx_desc40->txdw8'
add_force = ('\tif (tx_info->flags & IEEE80211_TX_CTL_INJECTED)\n'
             '\t\ttx_desc40->txdw3 |= cpu_to_le32(TXDESC40_USE_DRIVER_RATE);\n\n') + anchor_force

for anchor in (anchor_rate, anchor_force):
    if src.count(anchor) != 1:
        raise SystemExit(f'anchor not found exactly once: {anchor!r}')
src = src.replace(anchor_rate, add_rate).replace(anchor_force, add_force)
path.write_text(src)
print('patched', path)
