"""Where is ball-tracking performance lost?  (frame-level, centre distance; train = tuning split)
  1. oracle selection: GT frames with ANY candidate (score >= s_min) within r px  -> selection ceiling
  2. B0 threshold sweep: recall/precision of 'highest-scoring candidate if >= thr'
  3. miss-gap lengths of B0@0.35 (runs of GT frames without a correct output)
  4. B0 + linear interpolation of gaps <= L frames (image space and camera-motion-compensated)
  5. GT ball size stats and candidates per frame
Writes outputs/diag/ball_ceiling_<split>_<det>.json, prints a compact summary.
  python scripts/diag_ball_ceiling.py --split train --det yolov8m_repo
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.geometry.camera import apply_h
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--det", default="yolov8m_repo")
ap.add_argument("--r", type=float, default=10.0)
a = ap.parse_args(); R = a.r
seqs = sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
S_MIN = (0.05, 0.1, 0.2); THR = (0.05, 0.1, 0.2, 0.3, 0.35, 0.5, 0.7); GAPS = (5, 10, 25, 50)
acc = dict(n_gt=0, oracle={s: 0 for s in S_MIN}, sweep={t: [0, 0, 0] for t in THR}, gaps=[], interp={}, interp_cmc={},
           sizes=[], ncand=[], nearest_score=[])
for g in GAPS:
    acc["interp"][g] = [0, 0, 0]; acc["interp_cmc"][g] = [0, 0, 0]

def score_outputs(O, G, key, k):
    for f, o in O.items():
        acc[key][k][1] += 1
        if f in G and np.linalg.norm(o - G[f]) <= R: acc[key][k][2] += 1
    acc[key][k][0] += sum(1 for f, g in G.items() if f in O and np.linalg.norm(O[f] - g) <= R)

for s in seqs:
    seq = load_sequence(SNMOT_ROOT / a.split / s)
    gt = seq.gt(); gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
    G = {int(r[0]): r[2:4] + r[4:6] / 2 for r in gt}
    acc["sizes"] += list(np.sqrt(gt[:, 4] * gt[:, 5]))
    dets = detections_by_frame(np.loadtxt(ROOT / "cache/dets" / a.det / a.split / f"{s}.txt", delimiter=",", ndmin=2), seq.length)
    M = np.load(ROOT / "cache/snmot" / a.split / f"{s}.npz")["M"]
    balls = {f: (d[d[:, 5] == 3] if len(d) else d) for f, d in enumerate(dets)}
    acc["n_gt"] += len(G)
    for f, g in G.items():
        b = balls.get(f, np.zeros((0, 6)))
        acc["ncand"].append(len(b))
        if len(b):
            c = b[:, :2] + b[:, 2:4] / 2; dd = np.linalg.norm(c - g, axis=1); k = int(np.argmin(dd))
            if dd[k] <= R: acc["nearest_score"].append(float(b[k, 4]))
            for sm in S_MIN:
                acc["oracle"][sm] += bool(np.any((dd <= R) & (b[:, 4] >= sm)))
    for t in THR:
        O = {}
        for f in range(1, seq.length + 1):
            b = balls[f]
            if len(b) and b[:, 4].max() >= t:
                k = int(np.argmax(b[:, 4])); O[f] = b[k, :2] + b[k, 2:4] / 2
        score_outputs(O, G, "sweep", t)
        if t == 0.35:
            hits = sorted(f for f in O if f in G and np.linalg.norm(O[f] - G[f]) <= R)
            miss = [f for f in sorted(G) if f not in hits]
            run = 0; prev = None
            for f in miss:
                run = run + 1 if prev is not None and f == prev + 1 else 1
                if prev is not None and f != prev + 1: acc["gaps"].append(last_run)
                last_run = run; prev = f
            if miss: acc["gaps"].append(last_run)
            # interpolation of B0 outputs across gaps <= L (image space, and in a stabilised frame)
            fr = sorted(O)
            for L in GAPS:
                Oi, Oc = dict(O), dict(O)
                for f0, f1 in zip(fr, fr[1:]):
                    if 1 < f1 - f0 <= L + 1:
                        for f in range(f0 + 1, f1):
                            w = (f - f0) / (f1 - f0)
                            Oi[f] = (1 - w) * O[f0] + w * O[f1]
                            # CMC: carry both endpoints into frame f with the inter-frame homographies
                            p0 = O[f0].copy()
                            for q in range(f0 + 1, f + 1): p0 = apply_h(M[q], p0[None])[0] if M[q] is not None and np.all(np.isfinite(M[q])) else p0
                            p1 = O[f1].copy()
                            for q in range(f1, f, -1):
                                Mi = np.linalg.inv(M[q]) if M[q] is not None and np.all(np.isfinite(M[q])) else np.eye(3)
                                p1 = apply_h(Mi, p1[None])[0]
                            Oc[f] = (1 - w) * p0 + w * p1
                score_outputs(Oi, G, "interp", L); score_outputs(Oc, G, "interp_cmc", L)

n = acc["n_gt"]; pct = lambda x: round(100 * x / max(1, n), 1)
gaps = np.array(acc["gaps"]); sz = np.array(acc["sizes"])
summary = {
    "gt_ball_frames": n,
    "oracle_selection_recall@r": {str(k): pct(v) for k, v in acc["oracle"].items()},
    "B0_sweep(thr: recall, precision)": {str(t): (pct(v[0]), round(100 * v[2] / max(1, v[1]), 1)) for t, v in acc["sweep"].items()},
    "B0@0.35 + interp gaps<=L (recall, precision)": {str(L): (pct(v[0]), round(100 * v[2] / max(1, v[1]), 1)) for L, v in acc["interp"].items()},
    "B0@0.35 + CMC interp gaps<=L (recall, precision)": {str(L): (pct(v[0]), round(100 * v[2] / max(1, v[1]), 1)) for L, v in acc["interp_cmc"].items()},
    "miss_gap_frames(p50,p75,p90,max)": [int(np.percentile(gaps, q)) for q in (50, 75, 90)] + [int(gaps.max())] if len(gaps) else [],
    "frac_missed_frames_in_gaps<=10": round(float(gaps[gaps <= 10].sum() / max(1, gaps.sum())), 3) if len(gaps) else 0,
    "gt_ball_size_px(p10,p50,p90)": [round(float(np.percentile(sz, q)), 1) for q in (10, 50, 90)],
    "candidates_per_gt_frame(mean, frac0)": [round(float(np.mean(acc["ncand"])), 2), round(float(np.mean(np.array(acc["ncand"]) == 0)), 3)],
    "score_of_correct_candidate(p10,p50,p90)": [round(float(np.percentile(acc["nearest_score"], q)), 2) for q in (10, 50, 90)],
}
(ROOT / "outputs/diag").mkdir(parents=True, exist_ok=True)
(ROOT / f"outputs/diag/ball_ceiling_{a.split}_{a.det}.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items(): print(f"{k}: {v}")
