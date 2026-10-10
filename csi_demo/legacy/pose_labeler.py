#!/usr/bin/env python3
"""Mac-side 2D pose labeler.

live:    python pose_labeler.py --camera 0 --out DIR --duration 30 [--fps 30] [--preview]
offline: python pose_labeler.py --video FILE --timestamps FILE.npy --out DIR

Output (DIR): video.mp4 + frames_t.npy (live only), labels.npz, meta.json.
labels.npz: t_mac float64 (F,), keypoints float32 (F,P_max,14,3) [x_norm,y_norm,conf],
num_person int16 (F,), image_size int32 (2,) [width,height].

14-keypoint order: 0 neck, 1 head, 2 l_shoulder, 3 r_shoulder, 4 l_elbow, 5 l_hip,
6 r_elbow, 7 r_hip, 8 l_hand, 9 l_knee, 10 r_hand, 11 r_knee, 12 l_ankle, 13 r_ankle.
"""
import argparse
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# COCO-17 index for each output slot; -1 marks the neck (shoulder midpoint).
COCO_FOR_SLOT = [-1, 0, 5, 6, 7, 11, 8, 12, 9, 13, 10, 14, 15, 16]
MODEL_CANDIDATES = ("yolo11n-pose.pt", "yolov8n-pose.pt")


def load_model():
    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    last = None
    for name in MODEL_CANDIDATES:
        try:
            return YOLO(name), name, device
        except Exception as e:  # download/load failure -> try fallback
            last = e
    raise RuntimeError(f"could not load any pose model: {last}")


def coco17_to_14(kp, width, height):
    """kp: (17,3) pixel x, y, conf -> (14,3) normalized x, y in [0,1], conf."""
    out = np.zeros((14, 3), np.float32)
    for slot, idx in enumerate(COCO_FOR_SLOT):
        if idx >= 0:
            out[slot] = kp[idx]
        else:
            out[slot, :2] = (kp[5, :2] + kp[6, :2]) / 2
            out[slot, 2] = min(kp[5, 2], kp[6, 2])
    out[:, 0] /= width
    out[:, 1] /= height
    out[:, :2] = np.clip(out[:, :2], 0.0, 1.0)
    return out


def estimate(model, device, frame):
    """Return list of (14,3) arrays, one per detected person."""
    h, w = frame.shape[:2]
    res = model.predict(frame, device=device, verbose=False)[0]
    if res.keypoints is None or res.keypoints.data.shape[0] == 0:
        return []
    data = res.keypoints.data.cpu().numpy()  # (N,17,3)
    return [coco17_to_14(p, w, h) for p in data]


def pack_labels(t_mac, per_frame, size):
    f = len(per_frame)
    p_max = max([len(p) for p in per_frame] + [1])
    kps = np.zeros((f, p_max, 14, 3), np.float32)
    num = np.zeros(f, np.int16)
    for i, persons in enumerate(per_frame):
        num[i] = len(persons)
        for j, k in enumerate(persons):
            kps[i, j] = k
    return dict(
        t_mac=np.asarray(t_mac, np.float64),
        keypoints=kps,
        num_person=num,
        image_size=np.asarray(size, np.int32),
    )


def write_outputs(out, labels, meta):
    np.savez(out / "labels.npz", **labels)
    meta = dict(meta, created=datetime.now().isoformat(timespec="seconds"),
                num_frames=int(len(labels["t_mac"])))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def run_live(args, out):
    model, model_name, device = load_model()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    q = queue.Queue()  # unbounded so no frame is ever dropped from labeling
    results = {}

    def worker():
        while True:
            item = q.get()
            if item is None:
                return
            i, frame = item
            results[i] = estimate(model, device, frame)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    ts = []
    t_end = time.time() + args.duration
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            t = time.time()  # stamp right after grab
            if not ok:
                continue
            ts.append(t)
            writer.write(frame)
            q.put((len(ts) - 1, frame))
            if args.preview:
                cv2.imshow("pose_labeler", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        q.put(None)
        th.join()
        if args.preview:
            cv2.destroyAllWindows()
    np.save(out / "frames_t.npy", np.asarray(ts, np.float64))
    per_frame = [results[i] for i in range(len(ts))]
    labels = pack_labels(ts, per_frame, (w, h))
    dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    meta = dict(mode="live", camera=args.camera, fps=args.fps,
                measured_fps=(len(ts) - 1) / dur if dur > 0 else None,
                model=model_name, device=device)
    write_outputs(out, labels, meta)
    print(f"labeled {len(ts)} frames -> {out}")


def run_offline(args, out):
    t = np.load(args.timestamps).astype(np.float64).ravel()
    model, model_name, device = load_model()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    per_frame = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        per_frame.append(estimate(model, device, frame))
    cap.release()
    if len(per_frame) != len(t):
        n = min(len(per_frame), len(t))
        print(f"warning: {len(per_frame)} frames vs {len(t)} timestamps; truncating to {n}")
        per_frame, t = per_frame[:n], t[:n]
    labels = pack_labels(t, per_frame, (w, h))
    meta = dict(mode="offline", video=str(args.video), timestamps=str(args.timestamps),
                fps=video_fps, model=model_name, device=device)
    write_outputs(out, labels, meta)
    print(f"labeled {len(t)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int)
    ap.add_argument("--video")
    ap.add_argument("--timestamps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.video:
        if not args.timestamps:
            ap.error("--video requires --timestamps")
        run_offline(args, out)
    elif args.camera is not None:
        if not args.duration:
            ap.error("--camera requires --duration")
        run_live(args, out)
    else:
        ap.error("give --camera (live) or --video (offline)")


if __name__ == "__main__":
    main()
