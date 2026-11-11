"""What does a fast ball look like? Crops of GT balls binned by image speed (px/frame), and blur statistics:
for each crop, the second-moment ellipse of the ball-vs-background difference -> elongation and its
alignment with the motion direction.  Writes outputs/diag/fast_ball_montage.jpg and fast_ball_stats.json
  python scripts/viz_fast_ball.py --split train"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT, load_sequence
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--per-bin", type=int, default=8); a = ap.parse_args()
BINS = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 200)]
rng = np.random.default_rng(0); samples = {b: [] for b in BINS}; stats = {b: [] for b in BINS}
for s in sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir()):
    seq = load_sequence(SNMOT_ROOT / a.split / s); gt = seq.gt(); gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
    G = {int(r[0]): r[2:6] for r in gt}
    for f, g in G.items():
        if f - 1 not in G or f + 1 not in G: continue
        c = lambda b: b[:2] + b[2:] / 2
        v = (c(G[f + 1]) - c(G[f - 1])) / 2; sp = float(np.linalg.norm(v))
        for b in BINS:
            if b[0] <= sp < b[1]: samples[b].append((s, f, g, v, sp))
for b in BINS:
    pick = [samples[b][i] for i in rng.choice(len(samples[b]), min(len(samples[b]), 60), replace=False)] if samples[b] else []
    for s, f, g, v, sp in pick:
        img = cv2.imread(str(load_sequence(SNMOT_ROOT / a.split / s).img_path(f)))
        cx, cy = g[:2] + g[2:] / 2; r = 24
        x0, y0 = int(max(0, cx - r)), int(max(0, cy - r)); crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float32)
        if crop.shape[:2] != (2 * r, 2 * r): continue
        # ball-vs-background: deviation from the crop's median colour, weighted second moments
        dev = np.linalg.norm(crop - np.median(crop.reshape(-1, 3), 0), axis=2); w = np.clip(dev - np.percentile(dev, 70), 0, None)
        yy, xx = np.mgrid[:2 * r, :2 * r]; W = w.sum() + 1e-9; mx, my = (w * xx).sum() / W, (w * yy).sum() / W
        C = np.array([[(w * (xx - mx) ** 2).sum(), (w * (xx - mx) * (yy - my)).sum()], [(w * (xx - mx) * (yy - my)).sum(), (w * (yy - my) ** 2).sum()]]) / W
        ev, evec = np.linalg.eigh(C); elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        align = float(abs(evec[:, 1] @ (v / (np.linalg.norm(v) + 1e-9))))
        stats[b].append(dict(speed=sp, elong=elong, align=align, w=float(g[2]), h=float(g[3]), contrast=float(np.percentile(dev, 98))))
        if len([x for x in stats[b]]) <= a.per_bin:
            samples.setdefault(("img", b), []).append(cv2.resize(crop.astype(np.uint8), (144, 144), interpolation=cv2.INTER_NEAREST))
rows = []
for b in BINS:
    ims = samples.get(("img", b), [])[:a.per_bin]
    ims += [np.zeros((144, 144, 3), np.uint8)] * (a.per_bin - len(ims))
    row = np.hstack(ims); lab = np.zeros((26, row.shape[1], 3), np.uint8)
    S = stats[b]
    txt = f"speed {b[0]}-{b[1]} px/frame | n={len(samples[b])} | elongation p50 {np.median([x['elong'] for x in S]):.2f} | |cos(blur, motion)| p50 {np.median([x['align'] for x in S]):.2f} | box {np.median([x['w'] for x in S]):.0f}x{np.median([x['h'] for x in S]):.0f}px | contrast {np.median([x['contrast'] for x in S]):.0f}" if S else f"speed {b}: none"
    cv2.putText(lab, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); rows += [lab, row]
cv2.imwrite(str(ROOT / "outputs/diag/fast_ball_montage.jpg"), np.vstack(rows))
summ = {f"{b[0]}-{b[1]}": dict(n=len(samples[b]), **({k: float(np.median([x[k] for x in stats[b]])) for k in ("elong", "align", "w", "h", "contrast")} if stats[b] else {})) for b in BINS}
(ROOT / "outputs/diag/fast_ball_stats.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
