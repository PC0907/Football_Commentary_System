"""Detector recall (any ball candidate >= thr within 10 px) per GT ball image speed, plus share of GT frames.
  python scripts/diag_ball_recall_by_speed.py --det yolo26m_snmot --split train"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence
ap = argparse.ArgumentParser(); ap.add_argument("--det", default="yolo26m_snmot"); ap.add_argument("--split", default="train"); a = ap.parse_args()
BINS = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 1e9)]; THR = (0.05, 0.2)
cnt = {b: [0, {t: 0 for t in THR}, 0] for b in BINS}
for s in sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir()):
    seq = load_sequence(SNMOT_ROOT / a.split / s); gt = seq.gt(); gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
    C = {int(r[0]): r[2:4] + r[4:6] / 2 for r in gt}
    dets = detections_by_frame(np.loadtxt(ROOT / "cache/dets" / a.det / a.split / f"{s}.txt", delimiter=",", ndmin=2), seq.length)
    for f, c in C.items():
        if f - 1 in C and f + 1 in C: sp = np.linalg.norm(C[f + 1] - C[f - 1]) / 2
        elif f - 1 in C: sp = np.linalg.norm(C[f] - C[f - 1])
        else: continue
        b = next(x for x in BINS if x[0] <= sp < x[1]); cnt[b][0] += 1
        d = dets[f]; d = d[d[:, 5] == 3] if len(d) else d
        if len(d):
            dist = np.linalg.norm(d[:, :2] + d[:, 2:4] / 2 - c, axis=1)
            for t in THR: cnt[b][1][t] += bool(np.any((dist <= 10) & (d[:, 4] >= t)))
            cnt[b][2] += bool(np.any(dist <= 25))
tot = sum(v[0] for v in cnt.values())
out = {f"{b[0]}-{b[1] if b[1] < 1e8 else 'inf'} px/f": {"share": round(v[0] / tot, 3), **{f"recall@10 thr{t}": round(v[1][t] / max(1, v[0]), 3) for t in THR}, "any cand @25px": round(v[2] / max(1, v[0]), 3)} for b, v in cnt.items()}
(ROOT / f"outputs/diag/ball_recall_by_speed_{a.split}_{a.det}.json").write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(k, v)
