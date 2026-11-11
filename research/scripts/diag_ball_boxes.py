"""Is ball HOTA limited by box geometry? For detections whose centre is within 10 px of the GT ball:
IoU with the GT box, GT/det size ratio, centre offset.  Also: IoU if the box were re-sized by a single
global factor, or by the GT box size (upper bound of any box refinement).
  python scripts/diag_ball_boxes.py --det yolo26m_snmot --split train"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence
ap = argparse.ArgumentParser(); ap.add_argument("--det", default="yolo26m_snmot"); ap.add_argument("--split", default="train")
ap.add_argument("--thr", type=float, default=0.2); a = ap.parse_args()
def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0); return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)
def resize(b, s):
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2; return np.array([cx - b[2] * s / 2, cy - b[3] * s / 2, b[2] * s, b[3] * s])
R = []
for s in sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir()):
    seq = load_sequence(SNMOT_ROOT / a.split / s); gt = seq.gt(); gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
    G = {int(r[0]): r[2:6] for r in gt}
    dets = detections_by_frame(np.loadtxt(ROOT / "cache/dets" / a.det / a.split / f"{s}.txt", delimiter=",", ndmin=2), seq.length)
    for f, g in G.items():
        d = dets[f]; d = d[(d[:, 5] == 3) & (d[:, 4] >= a.thr)] if len(d) else d
        if not len(d): continue
        b = d[np.argmax(d[:, 4]), :4]
        off = np.linalg.norm(b[:2] + b[2:] / 2 - (g[:2] + g[2:] / 2))
        if off <= 10: R.append((iou(b, g), np.sqrt(g[2] * g[3]) / np.sqrt(b[2] * b[3]), off, np.sqrt(g[2] * g[3]), s))
R = np.array([r[:4] for r in R]); ious, ratio = R[:, 0], R[:, 1]
k = float(np.median(ratio))
out = {"n_matched_within_10px": len(R), "IoU(p10,p50,p90)": np.round(np.percentile(ious, [10, 50, 90]), 3).tolist(),
       "frac IoU>=0.5": round(float(np.mean(ious >= 0.5)), 3), "frac IoU>=0.75": round(float(np.mean(ious >= 0.75)), 3),
       "GT/det size ratio (p10,p50,p90)": np.round(np.percentile(ratio, [10, 50, 90]), 3).tolist(),
       "centre offset px (p50,p90)": np.round(np.percentile(R[:, 2], [50, 90]), 2).tolist(),
       "GT ball size px (p10,p50,p90)": np.round(np.percentile(R[:, 3], [10, 50, 90]), 1).tolist()}
(ROOT / "outputs/diag").mkdir(exist_ok=True); (ROOT / f"outputs/diag/ball_boxes_{a.split}_{a.det}.json").write_text(json.dumps(out, indent=1))
for k2, v in out.items(): print(f"{k2}: {v}")
