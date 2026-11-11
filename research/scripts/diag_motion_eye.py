"""Complementarity of the motion eye with the detector: per speed bin, recall@10 of YOLO (thr 0.2), of the motion
eye (full frame, top-5), and of their union; motion-eye false candidates per frame.
  python scripts/diag_motion_eye.py --seqs SNMOT-060 SNMOT-064"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence
from fcs.tracking.motion_eye import MotionEye
ap = argparse.ArgumentParser(); ap.add_argument("--seqs", nargs="+"); ap.add_argument("--split", default="train"); ap.add_argument("--det", default="yolo26m_snmot")
a = ap.parse_args()
BINS = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 1e9)]; acc = {b: np.zeros(4) for b in BINS}; fp = []; 
for s in a.seqs:
    seq = load_sequence(SNMOT_ROOT / a.split / s); gt = seq.gt(); gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
    C = {int(r[0]): r[2:4] + r[4:6] / 2 for r in gt}
    dets = detections_by_frame(np.loadtxt(ROOT / "cache/dets" / a.det / a.split / f"{s}.txt", delimiter=",", ndmin=2), seq.length)
    eye = MotionEye(seq, np.load(ROOT / "cache/snmot" / a.split / f"{s}.npz")["M"], dets)
    for f in range(1, seq.length + 1):
        m = eye(f)
        if f not in C or f - 1 not in C: continue
        sp = np.linalg.norm(C[f] - C[f - 1]); b = next(x for x in BINS if x[0] <= sp < x[1])
        d = dets[f]; d = d[(d[:, 5] == 3) & (d[:, 4] >= 0.2)] if len(d) else d
        hy = bool(len(d)) and bool(np.any(np.linalg.norm(d[:, :2] + d[:, 2:4] / 2 - C[f], axis=1) <= 10))
        hm = any(np.hypot(c[0] - C[f][0], c[1] - C[f][1]) <= 10 for c in m)
        acc[b] += [1, hy, hm, hy or hm]; fp.append(sum(np.hypot(c[0] - C[f][0], c[1] - C[f][1]) > 10 for c in m))
out = {f"{b[0]}-{b[1] if b[1] < 1e8 else 'inf'}": dict(n=int(v[0]), yolo=round(v[1] / max(1, v[0]), 3), motion=round(v[2] / max(1, v[0]), 3), union=round(v[3] / max(1, v[0]), 3)) for b, v in acc.items()}
out["motion_false_cands_per_frame"] = round(float(np.mean(fp)), 2)
print(json.dumps(out, indent=1))
