"""Kinematic plausibility of calibrations (GT-free calibration quality measure).

For every GT player track, the foot point is mapped through each frame's calibration and
the implied frame-to-frame ground speed is computed. Real players stay below ~10 m/s,
so the fraction of steps above 12 m/s ("violation rate") measures calibration error
(chiefly the common-mode jumps between independently fitted frames).

  python scripts/diag_calib_speed.py --modes c0 c1 fused smooth [--seqs ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.geometry.calib_provider import sequence_calibrations  # noqa: E402
from fcs.geometry.camera import apply_h  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT, load_sequence  # noqa: E402
from fcs.progress import Progress  # noqa: E402


def speeds(seq, calibs):
    gt = seq.gt()
    sp = []
    for tid in np.unique(gt[:, 1]):
        if seq.role_of(tid) == 3:
            continue
        r = gt[gt[:, 1] == tid]
        r = r[np.argsort(r[:, 0])]
        P = {}
        for row in r:
            H = calibs[int(row[0])][0]
            if H is not None:
                P[int(row[0])] = apply_h(H, [[row[2] + row[4] / 2, row[3] + row[5]]])[0]
        fr = sorted(P)
        for a, b in zip(fr, fr[1:]):
            if b == a + 1:
                sp.append(np.linalg.norm(P[b] - P[a]) * seq.fps)
    return np.array(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", nargs="+", default=["c0", "c1", "fused", "smooth"])
    ap.add_argument("--seqs", nargs="*")
    a = ap.parse_args()
    cache = ROOT / "cache" / "snmot" / a.split
    seqs = a.seqs or sorted(p.stem for p in cache.glob("*.npz"))
    agg = {m: {"sp": [], "cov": []} for m in a.modes}
    prog = Progress(len(seqs) * len(a.modes), f"calibration plausibility {a.split}", "seq·mode")
    for name in seqs:
        seq = load_sequence(SNMOT_ROOT / a.split / name)
        c = np.load(cache / f"{name}.npz")
        for m in a.modes:
            cal = sequence_calibrations(m, c["kp"], c["M"], seq.width, seq.height)
            agg[m]["sp"].append(speeds(seq, cal))
            agg[m]["cov"].append(np.mean([cal[t][0] is not None for t in range(1, seq.length + 1)]))
            prog.update(1, desc=f"calibration plausibility {a.split} {name} {m}")
    prog.close()
    out = {}
    for m in a.modes:
        sp = np.concatenate(agg[m]["sp"])
        out[m] = dict(coverage=float(np.mean(agg[m]["cov"])), p50=float(np.percentile(sp, 50)),
                      p90=float(np.percentile(sp, 90)), viol12=float(np.mean(sp > 12)),
                      n_steps=int(len(sp)))
        print(f"{m:7s} coverage {out[m]['coverage']:.3f}  speed p50 {out[m]['p50']:5.2f}  "
              f"p90 {out[m]['p90']:7.2f} m/s  violation(>12 m/s) {100 * out[m]['viol12']:5.2f}%  "
              f"[{len(seqs)} seqs, {len(sp)} steps]")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / f"calib_plausibility_{a.split}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
