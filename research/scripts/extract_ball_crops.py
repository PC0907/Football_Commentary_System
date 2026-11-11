"""Extract fovea training samples from SNMOT sequences (causal frame stacks around a simulated prediction).

Each sample = crops of frames (t-2, t-1, t) at the SAME window (native resolution, CROP×CROP), centred on a
simulated tracker prediction  μ = c_t + ε,  ε ~ N(0, σ²I),  σ ∈ {6, 16, 32, 48} px (σ is stored: it is the
prior's width, given to the network as an input at inference time — the tracker's predictive std).
Positives: GT ball visible at t.  Negatives (visibility 0): (a) ball absent/invisible at t, window at the last
seen position; (b) random windows ≥ 2·CROP/3 away from the ball.
Labels (crop coords at t): centre, box w/h, image velocity v_t = (c_{t+1} − c_{t−1})/2 (supervision only —
the network sees only t-2..t), visibility.

  python scripts/extract_ball_crops.py --snmot ROOT --split train --out balls.zip [--exclude-tuning --stride 2]
"""
import argparse
import configparser
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

CROP = 192
SIGMAS = (6, 16, 32, 48)


def seq_meta(sd: Path):
    s = configparser.ConfigParser(); s.read(sd / "seqinfo.ini"); s = s["Sequence"]
    g = configparser.ConfigParser(); g.read(sd / "gameinfo.ini")
    balls = {int(k.split("_")[1]) for k, v in g["Sequence"].items()
             if k.startswith("trackletid_") and v.strip().lower().startswith("ball")}
    return int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"]), balls


def window(c, W, H):
    x0 = int(np.clip(round(c[0] - CROP / 2), 0, W - CROP)); y0 = int(np.clip(round(c[1] - CROP / 2), 0, H - CROP))
    return x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snmot", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-tuning", action="store_true", help="drop every 4th sequence (the tracker/detector tuning split)")
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--neg-frac", type=float, default=0.3)
    ap.add_argument("--max-seqs", type=int); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    seqs = sorted(p for p in (Path(a.snmot) / a.split).iterdir() if p.is_dir())
    if a.exclude_tuning:
        tuning = set(p.name for p in seqs[::4]); seqs = [p for p in seqs if p.name not in tuning]
    seqs = seqs[:a.max_seqs] if a.max_seqs else seqs
    zf = zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED); labels = []; n_pos = n_neg = 0
    for si, sd in enumerate(seqs):
        L, W, H, ball_ids = seq_meta(sd)
        gt = np.loadtxt(sd / "gt" / "gt.txt", delimiter=",", ndmin=2)
        gt = gt[np.isin(gt[:, 1].astype(int), list(ball_ids))]
        C = {int(r[0]): (r[2:4] + r[4:6] / 2, r[4:6]) for r in gt}
        imgs = {}

        def frame(f):
            if f not in imgs:
                imgs[f] = cv2.imread(str(sd / "img1" / f"{f:06d}.jpg"))
                for k in [k for k in imgs if k < f - 3]:
                    del imgs[k]
            return imgs[f]

        last_seen = None
        for t in range(3, L + 1):
            if t in C:
                last_seen = C[t][0]
            if (t - 3) % a.stride:
                continue
            todo = []
            if t in C:
                sig = float(rng.choice(SIGMAS)); mu = C[t][0] + rng.normal(0, sig, 2)
                todo.append(("pos", mu, sig))
                if rng.random() < a.neg_frac:                     # random far window
                    for _ in range(10):
                        q = np.array([rng.uniform(0, W), rng.uniform(0, H)])
                        if np.linalg.norm(q - C[t][0]) > 2 * CROP / 3:
                            todo.append(("neg_far", q, float(rng.choice(SIGMAS)))); break
            elif last_seen is not None and rng.random() < 3 * a.neg_frac:   # ball invisible: look where it was
                sig = float(rng.choice(SIGMAS)); todo.append(("neg_absent", last_seen + rng.normal(0, sig, 2), sig))
            for kind, mu, sig in todo:
                x0, y0 = window(mu, W, H)
                crops = [frame(f)[y0:y0 + CROP, x0:x0 + CROP] for f in (t - 2, t - 1, t)]
                key = f"{sd.name}_{t:06d}_{kind}"
                for k, cimg in enumerate(crops):
                    ok, buf = cv2.imencode(".jpg", cimg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    zf.writestr(f"img/{key}_{k}.jpg", buf.tobytes())
                lab = dict(key=key, seq=sd.name, t=t, x0=x0, y0=y0, sigma=sig, prior=[float(mu[0] - x0), float(mu[1] - y0)], vis=0)
                if kind == "pos":
                    c, wh = C[t]
                    cx, cy = float(c[0] - x0), float(c[1] - y0)
                    if 0 <= cx < CROP and 0 <= cy < CROP:
                        v = None
                        if t - 1 in C and t + 1 in C:
                            v = ((C[t + 1][0] - C[t - 1][0]) / 2).tolist()
                        lab.update(vis=1, c=[cx, cy], wh=wh.tolist(), v=v)
                labels.append(lab); n_pos += lab["vis"]; n_neg += 1 - lab["vis"]
        print(f"[{si + 1}/{len(seqs)}] {sd.name}: total pos {n_pos} neg {n_neg}", flush=True)
    zf.writestr("labels.jsonl", "\n".join(json.dumps(l) for l in labels))
    zf.writestr("meta.json", json.dumps(dict(crop=CROP, sigmas=SIGMAS, seqs=[p.name for p in seqs], stride=a.stride)))
    zf.close()
    print("done", n_pos, n_neg)


if __name__ == "__main__":
    main()
