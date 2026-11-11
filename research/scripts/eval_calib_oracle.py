"""Oracle check of the camera solver + metric re-implementation.

Observations are built from the GROUND-TRUTH annotation points (total-least-squares line per
straight class, ellipse fit per circle class with >= 5 points), then solved with
fcs.calib.solve and scored with fcs.calib.sn_metric. This is an upper bound for any model
that predicts lines/ellipses (K2/K3) and a consistency test of conventions.

  python scripts/eval_calib_oracle.py --zip ~/soccerNet/hf/calib2023/valid.zip --n 300 [--no-priors]
"""
import argparse
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.calib.pitch import ALL_CLASSES, CIRCLE_CLASSES  # noqa: E402
from fcs.calib.sn_metric import H_EVAL, W_EVAL, evaluate_image, summarise  # noqa: E402
from fcs.calib.solve import Observations, solve_camera  # noqa: E402
from fcs.progress import Progress  # noqa: E402


def oracle_points(ann, sigma_px=1.5):
    """Point-level evidence: the annotated points themselves (upper bound for a model that
    outputs points on markings, e.g. the dense head)."""
    pts = {}
    for name, p in ann.items():
        if name in ALL_CLASSES and p:
            pts[name] = (np.array([[q["x"] * W_EVAL, q["y"] * H_EVAL] for q in p]), sigma_px)
    return Observations({}, {}, W_EVAL, H_EVAL, points=pts)


def oracle_obs(ann, sigma_px=1.5):
    lines, ells = {}, {}
    for name, pts in ann.items():
        if name not in ALL_CLASSES or not pts:
            continue
        P = np.array([[q["x"] * W_EVAL, q["y"] * H_EVAL] for q in pts])
        if name in CIRCLE_CLASSES:
            if len(P) >= 5:
                (cx, cy), (A, B), ang = cv2.fitEllipse(P.astype(np.float32))
                a, b, t = A / 2, B / 2, np.deg2rad(ang)
                c, s = np.cos(t), np.sin(t)
                Ra = np.array([[c, -s], [s, c]]); D = np.diag([1 / a ** 2, 1 / b ** 2])
                M = Ra @ D @ Ra.T; ctr = np.array([cx, cy])
                Cq = np.block([[M, (-M @ ctr)[:, None]], [(-M @ ctr)[None], np.array([[ctr @ M @ ctr - 1]])]])
                ells[name] = (Cq, sigma_px)
        elif len(P) >= 2:
            m = P.mean(0); _, _, Vt = np.linalg.svd(P - m)
            n = Vt[-1]
            lines[name] = (n, float(n @ m), sigma_px)
    return Observations(lines, ells, W_EVAL, H_EVAL)


def work(args):
    zp, name, priors, mode, dist = args
    z = zipfile.ZipFile(zp)
    ann = json.loads(z.read(name))
    obs = oracle_points(ann) if mode == "points" else oracle_obs(ann)
    sol = solve_camera(obs, priors=priors, distortion=dist)
    return {t: evaluate_image(sol.camera, ann, t) for t in (5, 10, 20)}, sol.camera is not None, \
        (None if sol.camera is None else float(-sol.camera.center[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(Path.home() / "soccerNet/hf/calib2023/valid.zip"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--mode", choices=["lines", "points"], default="lines")
    ap.add_argument("--distortion", action="store_true")
    a = ap.parse_args()
    with zipfile.ZipFile(a.zip) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json") and n.split("/")[-1][:5].isdigit())
    rng = np.random.default_rng(0)
    names = list(rng.choice(names, min(a.n, len(names)), replace=False))
    prog = Progress(len(names), f"oracle calib {a.mode} k1={a.distortion} priors={not a.no_priors}", "img")
    accs, heights = [], []
    with Pool(a.workers) as pool:
        for acc, ok, h in pool.imap_unordered(work, [(a.zip, n, not a.no_priors, a.mode, a.distortion) for n in names]):
            accs.append(acc); prog.update(1)
            if h is not None:
                heights.append(h)
    prog.close()
    res = summarise(accs)
    res["camera_height_median_m"] = float(np.median(heights)) if heights else None
    print(json.dumps(res, indent=1))
    tag = ("nopriors" if a.no_priors else "priors") + f"_{a.mode}" + ("_k1" if a.distortion else "")
    (ROOT / "outputs" / "pitchgeo").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "pitchgeo" / f"oracle_{tag}_{len(names)}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
