# build_2d_cache.py report

Script: `csi_demo/dataset/build_2d_cache.py`. It needs numpy and PyWavelets, which I installed into `csi_demo/.venv`.

## Usage
```
.venv/bin/python csi_demo/dataset/build_2d_cache.py --boards CSI-A,CSI-B,CSI-C \
  --session CSI_ROOT:LABELS_DIR [--session ...] --out DIR [--keep-empty] [--split-by-session NAME1,NAME2]
```
- The session name is the basename of `LABELS_DIR`. Frame names are `<session>_<frame index>`.
- `--split-by-session` writes `DIR/train` and `DIR/test`. Each is a full cache, and the named sessions go to test.

## Behaviour
- **Board time to Mac time:** the 32-bit `t_us` is unwrapped against `dev_now_us & 0xFFFFFFFF`, then `mac_t = header_mac_time - (dev_now_us - t_us_full)/1e6`.
- **Overlapping uploads:** records are de-duplicated across uploads by full board time, then sorted by `mac_t`.
- **Subcarriers:** `|real + j*imag|` with order -26..-1, +1..+26, linearly interpolated onto the 30 Intel-5300 indices. `np.interp` holds the edge values, so -28/28 clamp to the -26/26 values.
- **Window selection:** per frame at time T, take the 20 latest records with `mac_t <= T` on every board. The frame is skipped if the oldest of the 20 is older than T-0.5 s, or if `num_person == 0` (unless `--keep-empty`).
- **Smoothing and normalisation:** `waverec(wavedec(win, dB11, 'sym', axis=0))[:20]`, then divided by the window mean plus 1e-6.
- **Output:** `csi.npy` float32 (N, n_boards, 1, 20, 30), `keypoints.npy` float32 (N, P_max, 14, 2) zero-padded, `num_person.npy` int16, `names.npy`, and `meta.json`.
- **`meta.json`:** boards, counts, skipped counts by reason (`no_csi_window`, `no_person`), per-session info and time span.

## Fixture verification (synthetic, in /tmp/fx)
- **Fixture:** 3 boards with one npz each of 3000 records at 100 Hz. Board A's `t_us` wraps at 2^32. 600 label frames at 20 fps with 1-2 persons.
- **py_compile:** exit 0.
- **Converter output:** `[labels] kept 587/600 frames, skipped {'no_csi_window': 13, 'no_person': 0}`.
- **Shapes and dtypes:** csi `(587, 3, 1, 20, 30)`, keypoints `(587, 2, 14, 2)`, num_person `(587,)`, int16.
- **Window means:** between 0.9999998 and 1.0000002.
- **Assertions:** N > 500 and all values finite. The checker printed `PASS`.
- **Skipped frames:** the 13 skipped frames come after the last CSI record plus 0.5 s.
- **Split mode:** `--split-by-session labels` wrote the test split with 587 frames. Train came out empty (0 frames), which is expected when every session is in test.

## Notes
- The wavelet level for a 20-sample window with dB11 is 0 in pywt, so the smoothing is close to an identity. I followed the spec as written. Check that this matches your training pipeline.
- Not tested: real hardware data, an empty split (the converter still writes an empty train directory), and loading the cache through `WifiPoseCachedDataset`.
- The Python LSP is not installed here, so there are no LSP diagnostics. I used py_compile and the fixture run instead.
