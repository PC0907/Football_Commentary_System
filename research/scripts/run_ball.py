"""Run ball trackers B0..B3 on SNMOT with real detections and evaluate.

Metrics (ball GT only):
  * HOTA / DetA / AssA (TrackEval, ball-only GT)
  * frame-level localisation: recall@r = fraction of GT-ball frames with an output whose
    centre is within r px of the GT centre; precision@r = fraction of output frames
    within r px of a GT ball. r in {10, 25}.

  python scripts/run_ball.py --split train --det yolov8m_repo --variants B0 B1 B2 B3 --calib smoothL
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.geometry.calib_provider import sequence_calibrations  # noqa: E402
from fcs.tracking.ball import BallConfig, make_ball_tracker  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence, write_mot  # noqa: E402
from fcs.progress import Progress  # noqa: E402

BALL_ID = 1


def run_one(name, split, variant, det, calib, out_root, overrides):
    seq = load_sequence(SNMOT_ROOT / split / name)
    rows = np.loadtxt(ROOT / "cache" / "dets" / det / split / f"{name}.txt", delimiter=",", ndmin=2)
    dets = detections_by_frame(rows, seq.length)
    c = np.load(ROOT / "cache" / "snmot" / split / f"{name}.npz")
    cal = sequence_calibrations(calib, c["kp"], c["M"], seq.width, seq.height) if variant in ("B2", "B3", "B5", "B6") \
        else [(None, np.inf)] * (seq.length + 1)
    if variant.startswith("O"):                            # causal multi-eye trackers (fcs/tracking/ball_online.py)
        from fcs.tracking.ball_online import OnlineBallConfig, run_online
        EYES = {"O1": (("periph",), "never"), "O2": (("periph", "sahi"), "lost"), "O3": (("periph", "sahi"), "always"),
                "O4": (("periph", "motion"), "never"), "O5": (("periph", "sahi", "motion"), "lost"),
                "O6": (("periph", "fovea"), "never"), "O7": (("periph", "sahi", "fovea"), "lost"),
                "O8": (("periph", "sahi", "motion", "fovea"), "lost")}
        eyes, pol = EYES[variant]
        ocfg = OnlineBallConfig(eyes=eyes, sahi_policy=pol)
        for k, val in overrides.items():
            if "." in k:
                d, e = k.split("."); getattr(ocfg, d)[e] = val
            else:
                setattr(ocfg, k, val)
        balls = lambda dd: [np.zeros((0, 6))] + [x[x[:, 5] == 3] if len(x) else x for x in dd[1:]]
        static = {"periph": balls(dets)}
        if "sahi" in eyes:
            sp = ROOT / "cache" / "dets" / (det + "_sahi") / split / f"{name}.txt"
            sp = sp if sp.exists() else ROOT / "cache" / "dets" / "yolo26m_sahi" / split / f"{name}.txt"
            static["sahi"] = balls(detections_by_frame(np.loadtxt(sp, delimiter=",", ndmin=2), seq.length))
        online = {}
        if "motion" in eyes:
            from fcs.tracking.motion_eye import MotionEye
            online["motion"] = MotionEye(seq, c["M"], dets)
        if "fovea" in eyes:
            from fcs.ball.fovea_eye import FoveaEye
            online["fovea"] = FoveaEye(seq, overrides.get("fovea_ckpt", str(ROOT / "outputs/ball/fovea/best.pt")))
        rows_out = run_online(seq.length, static, ocfg, online)
        write_mot(out_root / f"{name}.txt", [(r[0], BALL_ID, *r[1:5], r[5]) for r in rows_out])
        return name
    if variant in ("B4", "B5", "B6"):                      # offline whole-clip trackers
        import cv2
        from fcs.tracking.ball_offline import OfflineBallConfig, track_offline
        ocfg = OfflineBallConfig(variant=variant, **overrides)
        balls = [np.zeros((0, 6))] + [d[d[:, 5] == 3] if len(d) else d for d in dets[1:]]
        rows_out = track_offline(seq.length, balls, c["M"], cal, ocfg, read_img=lambda f: cv2.imread(str(seq.img_path(f))))
        write_mot(out_root / f"{name}.txt", [(r[0], BALL_ID, *r[1:5], r[5]) for r in rows_out])
        return name
    cfg = BallConfig(variant=variant, fps=seq.fps, img_w=seq.width, img_h=seq.height, **overrides)
    trk = make_ball_tracker(cfg)
    out = []
    for f in range(1, seq.length + 1):
        d = dets[f]
        balls = d[d[:, 5] == 3] if len(d) else d
        players = d[(d[:, 5] <= 1) & (d[:, 4] >= 0.4)] if len(d) else d
        r = trk.step(f, balls, M=c["M"][f], H=cal[f][0], players=players)
        if r is not None:
            out.append((r[0], BALL_ID, *r[1:5], r[5]))
    write_mot(out_root / f"{name}.txt", out)
    return name


def loc_metrics(split, seqs, out_root):
    tp = {10: 0, 25: 0}; n_gt = 0; n_out = 0; tp_out = {10: 0, 25: 0}
    for name in seqs:
        seq = load_sequence(SNMOT_ROOT / split / name)
        gt = seq.gt()
        gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
        G = {int(r[0]): r[2:4] + r[4:6] / 2 for r in gt}
        p = out_root / f"{name}.txt"
        pr = np.loadtxt(p, delimiter=",", ndmin=2) if p.stat().st_size else np.zeros((0, 7))
        O = {int(r[0]): r[2:4] + r[4:6] / 2 for r in pr}
        n_gt += len(G); n_out += len(O)
        for f, g in G.items():
            if f in O:
                d = np.linalg.norm(O[f] - g)
                for r in tp:
                    tp[r] += d <= r
        for f, o in O.items():
            if f in G:
                d = np.linalg.norm(o - G[f])
                for r in tp_out:
                    tp_out[r] += d <= r
    res = {}
    for r in tp:
        res[f"recall@{r}"] = 100 * tp[r] / max(1, n_gt)
        res[f"prec@{r}"] = 100 * tp_out[r] / max(1, n_out)
    res["gt_frames"] = n_gt; res["out_frames"] = n_out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--det", default="yolov8m_repo")
    ap.add_argument("--variants", nargs="+", default=["B0", "B1", "B2", "B3"])
    ap.add_argument("--calib", default="smoothL")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], help="BallConfig overrides k=v")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    def _val(v):
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v
    overrides = {k: _val(v) for k, v in (s.split("=") for s in a.set)}
    seqs = a.seqs or sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
    from fcs.tracking.evaluate import evaluate
    dirs, summary = {}, {}
    prog = Progress(len(seqs) * len(a.variants), f"ball {a.split}/{a.det}", "seq")
    for v in a.variants:
        out_root = ROOT / "outputs" / "ball" / a.split / a.det / f"{v}{a.tag}"
        out_root.mkdir(parents=True, exist_ok=True)
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(partial(run_one, split=a.split, variant=v, det=a.det, calib=a.calib,
                                                 out_root=out_root, overrides=overrides), seqs):
                prog.update(1, desc=f"ball {a.split}/{a.det} variant {v}")
        dirs[f"{v}{a.tag}"] = out_root
    prog.close()
    res = evaluate(dirs, split=a.split, seqs=seqs, roles=[3], work=ROOT / "cache" / f"evalball_{a.split}")
    for v, r in res.items():
        r.update(loc_metrics(a.split, seqs, dirs[v]))
        summary[v] = r
        print(f"{v:8s} HOTA {r['HOTA']:5.1f} DetA {r['DetA']:5.1f} AssA {r['AssA']:5.1f} | "
              f"R@10 {r['recall@10']:5.1f} P@10 {r['prec@10']:5.1f} R@25 {r['recall@25']:5.1f} P@25 {r['prec@25']:5.1f}",
              flush=True)
    (ROOT / "outputs" / "ball" / a.split / a.det / f"summary{a.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
