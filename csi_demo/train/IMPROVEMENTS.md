# Fallback "v2" training plan (use if the current 9-link amplitude run stalls or overfits)

Baseline to beat (held-out s3, top-1 pose, `eval_room1.py`): epoch 5 = err 13.05%, pck@0.1 0.543
(mean-pose 15.52% / 0.304, shuffled-CSI 16.65% / 0.255).

Observed failure modes (epoch 5, GIFs in data/*pred_vs_gt*.gif):
- left-right position off by up to a body-width on unseen s3;
- distinctive arm poses (hands on head, arms spread) and steps/crossed legs mostly missed;
- prediction jumps between positions frame to frame;
- train/test gap: ~9-10% error on seen s1 clips vs 13% on unseen s3 -> some memorization of sessions.

All items below reuse s1-s3; none needs new recordings except the last group.

## Input
1. **Sanitized phase** (paper's `CSI_sanitization`, single-antenna form): per packet unwrap phase over the
   subcarriers, remove the least-squares line (slope + offset), keep the residual. Packet = 30 amplitude +
   30 phase -> `csi_in_channels=60` (the paper's setting).
2. **Gain-aware amplitude**: compensate each record with its stored `agc`/`fft` gain before the per-window
   mean normalization, so ESP32 AGC jumps stop looking like motion.
3. **Link dropout**: randomly zero one link's 20 packets during training (p~0.1) so the model does not lean
   on the strongest links; also makes it robust to a receiver dropping out.

## Labels / loss
4. **Mask unreliable keypoints**: zero the loss for joints with MediaPipe visibility < 0.5 or clipped to the
   image edge (ankles/knees in the current camera framing).
5. **Temporal label smoothing**: 5-frame (~0.2 s) moving average of keypoints to remove MediaPipe jitter.

## Model / training
6. **Drop the unused image backbone/neck** from the config (29M dead params; smaller checkpoint, nothing else).
7. **Train longer with cosine LR + warmup**, keep the best checkpoint by s3 pck@0.1 (not the last one).
8. **Temporal context at inference**: smooth predictions over ~5 frames (or feed 2 consecutive windows) to
   stop the skeleton jumping between positions.
9. **Against the train/test gap**: stronger weight decay / dropout, small amplitude noise and packet-shift
   augmentation, early stopping on s3 pck@0.1; optionally rotate the held-out session (train s1+s3, test s2,
   etc.) to check the result is not specific to s3.

## Data (only if the above plateaus)
10. More sessions on different days, camera moved back so ankles are visible, RTL dongles placed so their
   links cross the area (the 3 weak links run at ~35-50 of 100 packets/s).

Evaluate every variant with the same split and `eval_room1.py`; adopt a change only if it beats the
current best on s3.
