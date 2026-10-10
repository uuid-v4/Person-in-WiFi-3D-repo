#!/bin/sh
# Build and install the rate-injection-patched mt76 driver for the running Raspberry Pi kernel.
# Originals stay untouched; revert with: sudo rm -r /lib/modules/$(uname -r)/updates/csi-mt76 && sudo depmod -a
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
K=$(uname -r)
BRANCH="rpi-$(echo "$K" | cut -d. -f1,2).y"
sudo apt-get install -y build-essential git linux-headers-rpi-2712
SRC=$HOME/mt76src/linux-$BRANCH
[ -d "$SRC/.git" ] || git clone --depth 1 --filter=blob:none --sparse -b "$BRANCH" https://github.com/raspberrypi/linux.git "$SRC"
git -C "$SRC" sparse-checkout set drivers/net/wireless/mediatek/mt76
if git -C "$SRC" apply --reverse --check "$HERE/mt7921-rate-inject.patch" 2>/dev/null; then
  echo "patch already applied"
else
  git -C "$SRC" apply "$HERE/mt7921-rate-inject.patch"
fi
cd "$SRC/drivers/net/wireless/mediatek/mt76"
make -C /lib/modules/$K/build M=$PWD -j4 modules
D=/lib/modules/$K/updates/csi-mt76
for f in $(find . -name '*.ko'); do sudo install -D -m644 "$f" "$D/$f"; done
sudo depmod -a
for m in mt7921u mt7921_common mt792x_usb mt792x_lib mt76_connac_lib mt76_usb mt76; do sudo rmmod $m 2>/dev/null || true; done
sudo modprobe mt7921u
echo "patched mt7921u loaded: $(cat /sys/module/mt7921_common/srcversion)"
