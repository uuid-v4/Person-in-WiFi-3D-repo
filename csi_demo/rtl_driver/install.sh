#!/bin/sh
# Build and install the rate-injection-patched rtl8xxxu driver (RTL8192EU etc.) for the running Raspberry Pi kernel.
# Originals stay untouched; revert with: sudo rm /lib/modules/$(uname -r)/updates/rtl8xxxu.ko && sudo depmod -a
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
K=$(uname -r)
BRANCH="rpi-$(echo "$K" | cut -d. -f1,2).y"
sudo apt-get install -y build-essential git linux-headers-rpi-2712 python3
SRC=$HOME/mt76src/linux-$BRANCH
[ -d "$SRC/.git" ] || git clone --depth 1 --filter=blob:none --sparse -b "$BRANCH" https://github.com/raspberrypi/linux.git "$SRC"
git -C "$SRC" sparse-checkout add drivers/net/wireless/realtek/rtl8xxxu
D=$SRC/drivers/net/wireless/realtek/rtl8xxxu
python3 "$HERE/apply_patch.py" "$D/core.c"
make -C /lib/modules/$K/build M=$D CONFIG_RTL8XXXU=m -j4 modules
sudo install -D -m644 "$D/rtl8xxxu.ko" /lib/modules/$K/updates/rtl8xxxu.ko
sudo depmod -a
sudo modprobe -r rtl8xxxu 2>/dev/null || true
sudo modprobe rtl8xxxu
echo "patched rtl8xxxu loaded from: $(/sbin/modinfo -n rtl8xxxu 2>/dev/null || modinfo -n rtl8xxxu)"
