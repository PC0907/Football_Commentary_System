"""Break down oracle-calibration errors per image: which classes are false positives
(projected but not annotated), false negatives, or annotated-but-misaligned (max point
distance > thr). Writes outputs/pitchgeo/oracle_diag.json and prints a short summary."""
import json
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.calib.pitch import ALL_CLASSES, mirror_name  # noqa: E402
from fcs.calib.sn_metric import H_EVAL, W_EVAL, _dist_to_polyline, project_markings  # noqa: E402
from fcs.calib.solve import solve_camera  # noqa: E402
from fcs.progress import Progress  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from eval_calib_oracle import oracle_obs  # noqa: E402

ZIP = str(Path.home() / "soccerNet/hf/calib2023/valid.zip")


def work(name):
    z = zipfile.ZipFile(ZIP)
    ann = json.loads(z.read(name))
    sol = solve_camera(oracle_obs(ann))
    if sol.camera is None:
        return name, None
    gt = {k: np.array([[q["x"] * W_EVAL, q["y"] * H_EVAL] for q in v]) for k, v in ann.items() if k in ALL_CLASSES and v}
    pred = project_markings(sol.camera)
    best = None
    for g in (gt, {mirror_name(k): v for k, v in gt.items()}):
        fp = sorted(set(pred) - set(g)); fn = sorted(set(g) - set(pred))
        mis = {c: max(_dist_to_polyline(q, pred[c]) for q in g[c]) for c in set(pred) & set(g)}
        bad = {c: d for c, d in mis.items() if d >= 5}
        score = len(fp) + len(fn) + len(bad)
        if best is None or score < best["n_err"]:
            best = dict(n_err=score, fp=fp, fn=fn, misaligned=bad, n_common=len(mis),
                        median_maxdist=float(np.median(list(mis.values()))) if mis else None,
                        height=float(-sol.camera.center[2]), f=float(sol.camera.K[0, 0]))
    return name, best


if __name__ == "__main__":
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json") and n.split("/")[-1][:5].isdigit())
    names = list(np.random.default_rng(0).choice(names, 150, replace=False))
    prog = Progress(len(names), "oracle error breakdown", "img")
    rows = {}
    with Pool(12) as p:
        for n, r in p.imap_unordered(work, names):
            rows[n] = r; prog.update(1)
    prog.close()
    (ROOT / "outputs/pitchgeo/oracle_diag.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows.values() if r]
    fp = Counter(c for r in ok for c in r["fp"]); fn = Counter(c for r in ok for c in r["fn"])
    mis = Counter(c for r in ok for c in r["misaligned"])
    print(f"images {len(rows)}, solved {len(ok)}, perfect {sum(r['n_err'] == 0 for r in ok)}")
    print("median of per-image median max-dist (px):", np.median([r["median_maxdist"] for r in ok if r["median_maxdist"] is not None]))
    print("top FP classes:", fp.most_common(6))
    print("top FN classes:", fn.most_common(6))
    print("top misaligned classes:", mis.most_common(8))
    print("images with any FP:", sum(bool(r["fp"]) for r in ok), " any FN:", sum(bool(r["fn"]) for r in ok),
          " any misaligned:", sum(bool(r["misaligned"]) for r in ok))
