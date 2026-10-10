# Wi-Fi CSI 2D pose with ESP32 receivers (Person-in-WiFi 3D, low-cost build)

Fork of [Person-in-WiFi 3D](https://github.com/aiotgroup/Person-in-WiFi-3D-repo) (CVPR 2024); the original README is in [`PAPER_README.md`](PAPER_README.md). Our additions live in `csi_demo/`.

This fork adds a full capture -> dataset -> training pipeline to the Person-in-WiFi 3D code
(`opera/`, PETR transformer) using cheap hardware instead of Intel 5300 cards:

- **Transmitters (3):** Raspberry Pi 5 + Panda PAU0F (MT7921, tx1) and 2x RTL8192EU dongles (tx2, tx3),
  injecting 6 Mbps OFDM frames on channel 6 at 100 frames/s each.
- **Receivers (3):** ESP32-S3 boards measuring CSI (52 subcarriers) for every injected frame,
  streaming it over USB (one to the Mac, two to a second Pi that forwards over UDP).
- **Labels:** MacBook webcam + MediaPipe pose (14 keypoints, 2D + 3D).
- **Model:** the repo's PETR detector in 2D mode; input = 3 rx x 3 tx x 20 packets x 30 subcarriers.

Result (30 min of data, 1 person, train s1+s2, test on unseen s3, epoch 10, single most-confident pose):
joint error 12.5% of image, 54% of joints within 10% — vs 15.5% / 30% for always predicting the average
pose and 17.5% / 23% for the same model trained on shuffled (unpaired) CSI.

Held-out session s3 (never seen in training). Green = camera label, red = prediction from Wi-Fi CSI only
(epoch-5 model; first clip is one of the better stretches, second is typical):

![held-out, good stretch](csi_demo/docs/heldout_good.gif)
![held-out, typical stretch](csi_demo/docs/heldout_typical.gif)

The model tracks presence, height and overall posture and often the position; it still misses distinctive
arm poses and drifts left-right by up to a body width (see [`csi_demo/train/IMPROVEMENTS.md`](csi_demo/train/IMPROVEMENTS.md)).

## Physical setup (as recorded)

All devices at chest height on the corners/edges of a 1 m x 2 m rectangle; the person moves inside it.

```
                     top edge (1 m)
   RX  n16r8 ------------ Mac + webcam ------------ TX1  PAU0F on rpi1
   (USB-C to Mac)         (faces the area)              (MT7921, 6 Mbps, ch 6)
      |                                                    |
      |                                                    |
      |                 person moves here                  |   left / right
      |                  (labelled by the                  |   edges (2 m)
      |                   webcam pose model)               |
      |                                                    |
   RX  wroom2 ------------ rpi2 + TX2, TX3 ------------ RX  wroom1
   (USB to rpi2)       (2x RTL8192EU injecting)          (USB to rpi2)
                     bottom edge (1 m)
```

- 3 transmitters x 3 receivers = 9 links; each receiver tags every CSI record with the transmitter id
  (MACs 02:c5:1a:00:00:01..03), so all 9 links are separated in the data.
- Every receiver must see every transmitter; links that cross the area where the person moves carry the
  most information. In this placement the two RTLs share one spot, and the weakest links were
  n16r8 <- tx2/tx3 and wroom1 <- tx2 (~35-50 of 100 frames/s). Spreading the RTLs (e.g. mid-left and
  mid-right edges on USB extension cables) should help.
- Use the Pis' black USB 2.0 ports for the Wi-Fi dongles (USB 3.0 ports interfere with 2.4 GHz; the
  PAU0F hung on a blue port) and a 5 A supply for rpi2.
- Keep the camera far/high enough to see the whole body; in our recordings the ankles were cut off.

Data flow:

```
 rpi1: csi-tx (PAU0F, tx1) ~~~~\                      2.4 GHz ch 6, 6 Mbps OFDM, 100 frames/s each
 rpi2: csi-tx (RTL tx2, tx3) ~~~+~~> ESP32 receivers measure CSI per frame (52 subcarriers, I/Q)
                                        |
        n16r8 --USB------------------->  |  Mac: live/hub.py  <-- webcam (MediaPipe 2D+3D pose)
        wroom1, wroom2 --USB--> rpi2: forwarder.py --UDP:5599--> hub   (records data/sessions/<name>/)
                                        |
        dataset/build_2d_cache.py --> (N, 3 rx, 3 tx, 20 packets, 30 subcarriers) + keypoints
                                        |
        tools/train.py (PETR, 2D) on a GPU --> 14 keypoints per frame from Wi-Fi alone
```

## Layout

```
csi_demo/
  esp_ble_rx/      ESP32-S3 receiver firmware (ESP-IDF): CSI from tx ids 1-15, live USB stream + BLE buffer
  tx/              tx.py injector + csi-tx-all.sh (one injector per dongle, matched by MAC)
  pi/              systemd units and /etc/csi-tx.conf for the Pis
  pau0f_driver/    mt7921 patch + installer (honour radiotap rate on injected frames)
  rtl_driver/      rtl8xxxu patch (same for RTL8192EU)
  live/            hub.py (Mac): USB + UDP CSI ingest, webcam pose, session recorder, live page;
                   forwarder.py (Pi): USB CSI -> UDP; csi_frames.py: stream parser
  dataset/         build_2d_cache.py: sessions -> training cache (CSI windows + keypoints)
  infer/           predict.py (checkpoint -> poses), render_gif.py (prediction vs label overlay)
  train/           setup_wsl.sh (training env), eval_room1.py (strict held-out evaluation), IMPROVEMENTS.md (v2 plan), Wiception3D tools
  ble/             BLE collector + per-link check (fallback path, no USB)
  capture/         BLE-era session recorder (MediaPipe helpers reused by live/hub.py)
  legacy/          superseded experiments (serial dashboard, ESP-NOW tx, old inference)
configs/wifi/petr_wifi_esp_multitx_2d.py   training config for the 9-link dataset
```

Changes to the paper's code (`opera/`, `third_party/mmdet`): 2D keypoint mode (`coord_dim`), configurable
CSI token width (`csi_in_channels`), cached dataset loader, `pck2d` metric, faster keypoint match cost.

## What is and is not reproducible

- **Reproducible from this repo:** the whole pipeline — firmware, driver patches, transmitter/forwarder
  services, live hub + recorder, dataset builder, training config, evaluation — on equivalent hardware.
- **Not bit-exact:** the reported numbers come from our own recordings (one person, one room, 30 min),
  which are not published (they include video of the subject). With your own hardware you get your own
  dataset; expect the same *test* (real CSI beats mean-pose and shuffled-CSI baselines on a held-out
  session), not the same digits. Wi-Fi CSI depends on the room, device placement and the person.
- **Training randomness:** train with `--seed 0 --deterministic` for repeatable runs on one machine.
- **Tested versions:** Mac Python 3.12.12 (`csi_demo/requirements.txt`, pinned); ESP-IDF v5.5.1 (components pinned
  in `csi_demo/esp_ble_rx/dependencies.lock`); Raspberry Pi OS 13 (trixie), kernel 6.18.50+rpt-rpi-2712; training
  Python 3.8.20, PyTorch 1.13.1+cu117, MMCV 1.5.3 (built with CUDA ops), MMDetection 2.25.0 (vendored),
  NumPy 1.23.5, on an NVIDIA A100.
- **Setup-specific values you must change:**

  | What | Where |
  |---|---|
  | Transmitter dongle MACs | `csi_demo/pi/csi-tx.conf` |
  | Mac IP the receiver Pi forwards to | `csi_demo/pi/csi-forward.service` (`--hub`) |
  | Receiver board names by USB serial | `csi_demo/live/forwarder.py --name MAC=NAME` (or rename boards over BLE) |
  | Transmitter Pi IPs for the status panel | `csi_demo/live/hub.py --pi LABEL=IP` |
  | Webcam index | `csi_demo/live/hub.py --camera N` |
  | Board names / order in the dataset | `csi_demo/dataset/build_2d_cache.py --boards ...` |
  | Layout description saved per session | `LAYOUT` in `csi_demo/live/hub.py` |

## Reproduce

Not in git (regenerated or downloaded): `data/` (recordings, caches), `work_dirs/`, ESP-IDF `build/`,
`managed_components/`, `sdkconfig`, model weights (`*.task` is downloaded by the hub on first run).

End to end: **install (1-4) -> record with the dashboard (5) -> build dataset (6) -> train (7) -> infer (8)**.

### 1. Host / dashboard machine (macOS tested; Linux should work)
```bash
git clone <this repo> && cd Person-in-WiFi-3D-repo
python3.12 -m venv csi_demo/.venv
csi_demo/.venv/bin/pip install -r csi_demo/requirements.txt
```
macOS asks for camera access the first time `hub.py` opens the webcam: allow it for your terminal app.
`ssh` key access to the Pis (`ssh-copy-id sijin@<pi>`) is only needed for the status panel.

### 2. Receivers: flash the ESP32-S3 boards (ESP-IDF v5.5.1)
Install ESP-IDF once ([official guide](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/get-started/)):
```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.5.1 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3 && . ./export.sh
```
Then for each board (plug its **USB** port, not UART):
```bash
cd csi_demo/esp_ble_rx
idf.py set-target esp32s3            # first time only
idf.py -p /dev/cu.usbmodemXXXX flash  # Linux: /dev/ttyACM0
```
A board names itself `CSI-<last 3 MAC bytes>` (e.g. `CSI-298988`). Optional: rename by writing `NAME=wroom1`
to the BLE control characteristic (`c51a0002-...`, e.g. from nRF Connect); it saves to flash and reboots.
Boards on a Pi can also be flashed from the Pi:
`python3 -m esptool --chip esp32s3 -p /dev/serial/by-id/<board> --no-stub write_flash 0x0 bootloader.bin 0xa000 partition-table.bin 0x20000 esp_ble_rx.bin`
(files from `build/`).

### 3. Transmitter Pis: drivers + injector service
Raspberry Pi OS (64-bit, trixie tested), Pi 5. Plug the Wi-Fi dongles into the **black USB 2.0** ports.
Stock drivers send injected frames at 1 Mbps DSSS (no CSI on the ESP32); the patched ones honour the
radiotap rate (6 Mbps OFDM). Both installers fetch only the needed kernel driver folder for the running
kernel, build it out of tree, install it under `/lib/modules/$(uname -r)/updates/` and reload it
(originals untouched; the scripts print how to revert):
```bash
git clone <this repo> && cd Person-in-WiFi-3D-repo
bash csi_demo/pau0f_driver/install.sh    # Panda PAU0F / MT7921U
bash csi_demo/rtl_driver/install.sh      # RTL8192EU dongles (rtl8xxxu)
```
Re-run the installer after a kernel upgrade (`uname -r` changed), or hold the kernel:
`sudo apt-mark hold linux-image-rpi-2712`. Then the injector service:
```bash
sudo apt install -y python3-scapy
cp csi_demo/tx/tx.py ~/tx.py                        # csi-tx-all.sh expects /home/sijin/tx.py: edit TX= if your user differs
sudo install -m755 csi_demo/tx/csi-tx-all.sh /usr/local/bin/csi-tx-all
ip -br link                                         # note each dongle's MAC
sudo install -m644 csi_demo/pi/csi-tx.conf /etc/csi-tx.conf && sudo nano /etc/csi-tx.conf   # one line per dongle: id MAC 100
sudo install -m644 csi_demo/pi/csi-tx.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now csi-tx
journalctl -u csi-tx -f                             # expect "tx N: sent=... rate=100.0/s" per dongle
```

### 4. Receiver Pi (ESP32s on USB): forwarder service
```bash
sudo apt install -y python3   # stdlib only
cp csi_demo/live/forwarder.py csi_demo/live/csi_frames.py ~/
sudo nano csi_demo/pi/csi-forward.service    # set --hub <dashboard machine IP>, path if your user differs
sudo install -m644 csi_demo/pi/csi-forward.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now csi-forward
journalctl -u csi-forward -f                 # expect "CSI-xxxx=80/s ..." per board
```
(Boards plugged straight into the dashboard machine need no forwarder.)

### 5. Collect data with the dashboard
```bash
csi_demo/.venv/bin/python csi_demo/live/hub.py --camera 1 --pi tx-pi=<IP> --pi rx-pi=<IP>
# open http://localhost:8766/live
```
Check: every receiver x transmitter cell shows **LIVE** (~40-90 pkt/s), the skeleton follows you and the whole
body is in frame. Enter a session name + minutes, press **Start**, move around (walk, arms, turn, crouch,
pause); it saves automatically to `data/sessions/<name>/`. Record at least 3 x 10 min and keep one session for
testing. Keep every device in place between sessions.

### 6. Build the dataset
```bash
csi_demo/.venv/bin/python csi_demo/dataset/build_2d_cache.py --boards <rx1>,<rx2>,<rx3> \
  --tx-ids 1,2,3 --session data/sessions/s1/csi:data/sessions/s1/labels \
  --session data/sessions/s2/csi:data/sessions/s2/labels --session data/sessions/s3/csi:data/sessions/s3/labels \
  --out data/cache_9link --split-by-session s3
```
`--boards` are the receiver names shown on the dashboard; keep the same order for every build.

### 7. Train (Linux/WSL2 + NVIDIA GPU)
One-time environment (conda, PyTorch 1.13.1+cu117, MMCV 1.5.3 with CUDA ops, vendored MMDetection 2.25):
```bash
ARCH=8.6 bash csi_demo/train/setup_wsl.sh   # ARCH = GPU compute capability (8.0 A100, 8.6 RTX30, 8.9 RTX40)
conda activate piw3d
```
Copy `data/cache_9link/` to the GPU machine (same relative path), then:
```bash
PYTHONPATH=. python tools/train.py configs/wifi/petr_wifi_esp_multitx_2d.py --seed 0 --deterministic
PYTHONPATH=. python csi_demo/train/eval_room1.py configs/wifi/petr_wifi_esp_multitx_2d.py \
  work_dirs/petr_wifi_esp_multitx_2d/epoch_10.pth --cache data/cache_9link
```
50 epochs take ~3.5-7 h on one A100. For a control, train once more on a copy of the cache with
`train/csi.npy` rows permuted; the real model must beat it and the mean-pose baseline (both printed by
`eval_room1.py`).

### 8. Inference and visualisation
```bash
# GPU machine: predict the most confident pose for each window of a split
PYTHONPATH=. python csi_demo/infer/predict.py configs/wifi/petr_wifi_esp_multitx_2d.py \
  work_dirs/petr_wifi_esp_multitx_2d/epoch_10.pth --cache data/cache_9link --split test \
  --start 0 --count 300 --out preds.npz
# dashboard machine (has the session video): overlay prediction (red) vs camera label (green)
csi_demo/.venv/bin/python csi_demo/infer/render_gif.py preds.npz --cache data/cache_9link --split test --out clip.gif
```
**Live input for real-time inference:** while `hub.py` runs (add `--camera -1` to skip the webcam),
`GET http://localhost:8766/live/window?mode=v1` returns the newest model-ready window built from the last
20 packets of every link, preprocessed exactly like `build_2d_cache.py` (`mode=v2` adds sanitized phase
and per-packet normalisation, 60 values per packet; `&format=npy` returns a raw `.npy` instead of JSON).
It answers 503 until every link has 20 packets within the last 0.7 s. Feed it to the model at 2-5 Hz
(one window = `simple_test` input with a batch axis added) to get live poses. The window code is
`csi_demo/live/windowing.py`; replaying recorded sessions through it reproduces the training cache
exactly.
