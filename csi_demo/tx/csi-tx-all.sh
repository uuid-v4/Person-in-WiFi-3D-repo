#!/bin/bash
# Start one CSI injector per dongle listed in /etc/csi-tx.conf ("<tx_id> <mac> <frames_per_s>").
# Dongles are matched by MAC, so USB order / wlanN names do not matter; missing dongles are skipped.
set -u
CONF=${CONF:-/etc/csi-tx.conf}
TX=${TX:-/home/sijin/tx.py}
pids=()
while read -r id mac rate; do
    [[ -z "${id:-}" || "$id" == \#* ]] && continue
    iface=""
    for d in /sys/class/net/*; do
        [[ "$(cat "$d/address" 2>/dev/null)" == "$mac" ]] && iface=$(basename "$d")
    done
    if [[ -z "$iface" ]]; then
        echo "tx $id: $mac not present, skipping"
        continue
    fi
    ip link set "$iface" down
    iw dev "$iface" set type monitor
    ip link set "$iface" up
    iw dev "$iface" set channel 6 HT20
    echo "tx $id: $mac on $iface at $rate/s"
    python3 -u "$TX" --iface "$iface" --rate "$rate" --mbps 6 --tx-id "$id" 2>&1 | sed -u "s/^/tx $id: /" &
    pids+=($!)
done < "$CONF"
[[ ${#pids[@]} -eq 0 ]] && { echo "no transmitter dongles present"; exit 1; }
wait -n
exit 1
