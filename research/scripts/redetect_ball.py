"""Tracking-guided ball re-detection ("the track tells the detector where to look").

1. Run the offline path search (B4) on the existing detections.
2. For every frame where the path has no detection, predict where the ball should be:
   linear interpolation between the neighbouring path nodes (gap ≤ max_gap), else extrapolate
   from the nearest node within `extrap` frames.
3. Crop a window around the prediction, upscale ×`zoom` (a 15 px ball becomes 30 px — small objects
   are where detectors fail), run the same detector, map ball boxes back to full-frame coordinates.
4. Write a new detection cache = original detections + the new ball candidates (class 3), so every
   tracker can use it:  cache/dets/<det><tag>/<split>/<seq>.txt

  python scripts/redetect_ball.py --weights outputs/detector/yolo26m_snmot/det/train/weights/best.pt \
      --det yolo26m_snmot --split train [--zoom 2 --crop 320]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.progress import Progress  # noqa: E402
from fcs.tracking.ball_offline import OfflineBallConfig, track_offline  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence  # noqa: E402

BALL = 3


def predict_positions(path_rows, length, max_gap=50, extrap=12):
    """frame -> predicted centre for frames without a path node."""
    nodes = {int(r[0]): np.array([r[1] + r[3] / 2, r[2] + r[4] / 2]) for r in path_rows}
    fr = sorted(nodes); pred = {}
    for f0, f1 in zip(fr, fr[1:]):
        if 1 < f1 - f0 <= max_gap:
            for q in range(f0 + 1, f1):
                w = (q - f0) / (f1 - f0); pred[q] = (1 - w) * nodes[f0] + w * nodes[f1]
    for f in range(1, length + 1):
        if f in nodes or f in pred:
            continue
        near = [g for g in fr if abs(g - f) <= extrap]
        if near:
            pred[f] = nodes[min(near, key=lambda g: abs(g - f))]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--det", required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--tag", default="_redet")
    ap.add_argument("--zoom", type=float, default=2.0); ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.05); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seqs", nargs="*")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    seqs = a.seqs or sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
    out_dir = ROOT / "cache" / "dets" / f"{a.det}{a.tag}" / a.split; out_dir.mkdir(parents=True, exist_ok=True)
    prog = Progress(len(seqs), f"ball re-detection {a.split} (zoom {a.zoom})", "seq")
    stats = []
    for s in seqs:
        seq = load_sequence(SNMOT_ROOT / a.split / s)
        raw = np.loadtxt(ROOT / "cache" / "dets" / a.det / a.split / f"{s}.txt", delimiter=",", ndmin=2)
        dets = detections_by_frame(raw, seq.length)
        balls = [np.zeros((0, 6))] + [d[d[:, 5] == BALL] if len(d) else d for d in dets[1:]]
        no_cal = [(None, 0)] * (seq.length + 1)          # B4 needs no calibration; CMC is off for the ball
        path = track_offline(seq.length, balls, [None] * (seq.length + 1), no_cal, OfflineBallConfig(L=0))
        pred = predict_positions(path, seq.length)
        half = a.crop // 2; new_rows = []
        items = sorted(pred.items())
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]; crops, offs = [], []
            for f, c in chunk:
                img = cv2.imread(str(seq.img_path(f)))
                x0 = int(np.clip(c[0] - half, 0, seq.width - a.crop)); y0 = int(np.clip(c[1] - half, 0, seq.height - a.crop))
                crop = img[y0:y0 + a.crop, x0:x0 + a.crop]
                crops.append(cv2.resize(crop, None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_CUBIC)); offs.append((f, x0, y0))
            res = model(crops, imgsz=int(a.crop * a.zoom), conf=a.conf, verbose=False, half=True, classes=[BALL])
            for (f, x0, y0), r in zip(offs, res):
                for (bx0, by0, bx1, by1), sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    X0, Y0 = x0 + bx0 / a.zoom, y0 + by0 / a.zoom
                    new_rows.append([f, -1, X0, Y0, (bx1 - bx0) / a.zoom, (by1 - by0) / a.zoom, sc, BALL, -1, -1])
        allr = np.vstack([raw] + ([np.array(new_rows)] if new_rows else []))
        np.savetxt(out_dir / f"{s}.txt", allr, delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.4f", "%d", "%d", "%d"])
        stats.append((s, len(pred), len(new_rows))); prog.update(1)
    prog.close()
    for s, n, k in stats:
        print(f"{s}: looked in {n} frames, {k} new ball candidates")


if __name__ == "__main__":
    main()
