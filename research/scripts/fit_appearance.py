"""Fit the appearance sensor's likelihood ratio on the TRAIN split.

For GT tracks, compare torso colour histograms (Bhattacharyya coefficient rho) of
  same identity  : same id, frame gap 1..75
  different id   : different ids in the same frame (the realistic confusers)
and estimate log p(rho|same) - log p(rho|diff) with Gaussian KDEs on a grid, made
monotone non-decreasing (higher similarity can never be evidence *against* identity).
Output: cache/app_lr.json  {rho: [...], llr: [...]}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT, list_sequences  # noqa: E402
from fcs.tracking.tracker import bhattacharyya, torso_hist  # noqa: E402

rng = np.random.default_rng(0)
same, diff = [], []
for seq in list_sequences("train"):
    gt = seq.gt()
    frames = range(1, seq.length + 1, 15)
    H = {}
    for f in frames:
        img = cv2.imread(str(seq.img_path(f)))
        for r in gt[gt[:, 0] == f]:
            if seq.role_of(r[1]) == 3:
                continue
            h = torso_hist(img, r[2:6])
            if h is not None:
                H[(f, int(r[1]))] = h
    keys = list(H)
    by_frame = {}
    for f, i in keys:
        by_frame.setdefault(f, []).append(i)
    for f, ids in by_frame.items():
        for a in ids:
            for g in (15, 30, 60):
                if (f + g, a) in H:
                    same.append(bhattacharyya(H[(f, a)], H[(f + g, a)]))
            for b in ids:
                if b > a:
                    diff.append(bhattacharyya(H[(f, a)], H[(f, b)]))
    print(seq.name, len(same), len(diff), flush=True)

same, diff = np.array(same), np.array(diff)
grid = np.linspace(0, 1, 101)
ks, kd = gaussian_kde(same, 0.05), gaussian_kde(diff, 0.05)
llr = np.log(ks(grid) + 1e-6) - np.log(kd(grid) + 1e-6)
llr = np.maximum.accumulate(np.clip(llr, -6, 6))
out = ROOT / "cache" / "app_lr.json"
out.write_text(json.dumps({"rho": grid.tolist(), "llr": llr.tolist(),
                           "n_same": int(len(same)), "n_diff": int(len(diff)),
                           "mean_same": float(same.mean()), "mean_diff": float(diff.mean())}))
# separability summary
from sklearn.metrics import roc_auc_score  # noqa: E402
y = np.r_[np.ones(len(same)), np.zeros(len(diff))]
print("AUC same-vs-diff:", roc_auc_score(y, np.r_[same, diff]), "->", out)
