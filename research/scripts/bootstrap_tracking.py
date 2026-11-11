"""Paired significance tests between tracker variants, over sequences (already-run outputs).

  python scripts/bootstrap_tracking.py --split test --det oracle --base T1 X-bytetrack X-ocsort-cmc X-botsort --ours T4

For each (ours, base) pair and metric (HOTA, AssA, IDF1, IDSW) it reports:
  the mean per-sequence difference, a 95% paired-bootstrap CI (10,000 resamples of sequences),
  the Wilcoxon signed-rank p-value, and wins/ties/losses over sequences.
Output: outputs/tracking/<split>/<det>/significance_<ours>.json (+ printed table).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT  # noqa: E402
from fcs.tracking.evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test"); ap.add_argument("--det", default="oracle")
ap.add_argument("--ours", default="T4"); ap.add_argument("--base", nargs="+", required=True)
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()
seqs = sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
base_dir = ROOT / "outputs" / "tracking" / a.split / a.det
names = [a.ours] + a.base
res = evaluate({v: base_dir / v for v in names}, split=a.split, seqs=seqs, roles=[0, 1, 2, 4],
               work=ROOT / "cache" / f"eval_sig_{a.split}_{a.det}", per_seq=True)
rng = np.random.default_rng(0)
out = {"combined": {v: {k: x for k, x in res[v].items() if not k.startswith("_")} for v in names}, "pairs": {}}
print(f"{'pair':28s} {'metric':6s} {'mean diff':>9s} {'95% CI':>18s} {'Wilcoxon p':>11s}  W/T/L")
for b in a.base:
    out["pairs"][b] = {}
    for m in ("HOTA", "AssA", "IDF1", "IDSW"):
        d = np.array([res[a.ours]["_per_seq"][s][m] - res[b]["_per_seq"][s][m] for s in seqs], float)
        boot = d[rng.integers(0, len(d), (a.n_boot, len(d)))].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        better = d < 0 if m == "IDSW" else d > 0
        w, t = int(better.sum()), int((d == 0).sum())
        out["pairs"][b][m] = dict(mean=float(d.mean()), ci95=[float(lo), float(hi)], wilcoxon_p=p,
                                  wins=w, ties=t, losses=len(d) - w - t)
        print(f"{a.ours + ' vs ' + b:28s} {m:6s} {d.mean():9.2f} [{lo:7.2f},{hi:7.2f}] {p:11.2e}  {w}/{t}/{len(d) - w - t}")
(base_dir / f"significance_{a.ours}.json").write_text(json.dumps(out, indent=1))
