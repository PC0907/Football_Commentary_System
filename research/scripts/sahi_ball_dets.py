"""SAHI-style sliced ball detection (per frame, causal): YOLO on overlapping native-resolution tiles, ball class
only, merged by class-wise NMS.  Precomputed for every frame so the online tracker can decide WHEN to look
(e.g. only while the ball is lost — the 'saccade' policy); using a frame's tiles needs only that frame.
Output: cache/dets/<tag>/<split>/<seq>.txt  (same format; class 3 rows only)

  python scripts/sahi_ball_dets.py --weights .../best.pt --split train --tag yolo26m_sahi [--tile 640 --overlap 0.2]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.progress import Progress  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT, load_sequence  # noqa: E402


def tiles(W, H, t, ov):
    step = int(t * (1 - ov))
    xs = list(range(0, max(1, W - t) + 1, step)); ys = list(range(0, max(1, H - t) + 1, step))
    if xs[-1] != W - t: xs.append(W - t)
    if ys[-1] != H - t: ys.append(H - t)
    return [(x, y) for y in ys for x in xs]


def nms(b, s, thr=0.5):
    if not len(b):
        return []
    idx = cv2.dnn.NMSBoxes([list(map(float, x)) for x in b], list(map(float, s)), 0.0, thr)
    return np.array(idx).reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--tag", required=True)
    ap.add_argument("--tile", type=int, default=640); ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.05); ap.add_argument("--seqs", nargs="*")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    seqs = a.seqs or sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
    out = ROOT / "cache" / "dets" / a.tag / a.split; out.mkdir(parents=True, exist_ok=True)
    total = sum(load_sequence(SNMOT_ROOT / a.split / s).length for s in seqs)
    prog = Progress(total, f"SAHI ball detection {a.split} (tile {a.tile})", "frame")
    for s in seqs:
        seq = load_sequence(SNMOT_ROOT / a.split / s); T = tiles(seq.width, seq.height, a.tile, a.overlap); rows = []
        for f in range(1, seq.length + 1):
            img = cv2.imread(str(seq.img_path(f)))
            crops = [img[y:y + a.tile, x:x + a.tile] for x, y in T]
            res = model(crops, imgsz=a.tile, conf=a.conf, verbose=False, half=True, classes=[3])
            B, S = [], []
            for (x, y), r in zip(T, res):
                for (x0, y0, x1, y1), sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    B.append([x + x0, y + y0, x1 - x0, y1 - y0]); S.append(sc)
            for k in nms(B, S):
                x, y, w, h = B[k]; rows.append(f"{f},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{S[k]:.4f},3,-1,-1")
            prog.update(1)
        (out / f"{s}.txt").write_text("\n".join(rows) + "\n")
    prog.close()


if __name__ == "__main__":
    main()
