"""Run tracker variants on SNMOT and evaluate (HOTA/DetA/AssA/IDF1).

  python scripts/run_tracking.py --split test --variants T0 T1 T2 T3 T4 [--seqs ...] [--det oracle]

Results: outputs/tracking/<split>/<det>/<variant>/<seq>.txt and a JSON summary.
"""
import argparse
import json
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from fcs.geometry.calib_provider import sequence_calibrations  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT, load_sequence, oracle_detections, write_mot  # noqa: E402
from fcs.tracking.tracker import MultiTracker, TrackerConfig  # noqa: E402
from fcs.progress import Progress  # noqa: E402


def app_lr_from_file(path: Path):
    """Appearance log-likelihood ratio fitted on the tuning split (see fit_appearance.py)."""
    if not path.exists():
        return None
    p = json.loads(path.read_text())
    xs, ys = np.array(p["rho"]), np.array(p["llr"])
    return lambda r: np.interp(r, xs, ys)


def variant_config(name: str, fps: float, app_lr) -> tuple:
    """Returns (TrackerConfig, calibration mode)."""
    P = dict(motion="pitch", cmc=True, het_R=True, v_max=10.0, assoc="mahal", pitch_stage="lost")
    V = {
        "T1": (dict(motion="image", cmc=True, assoc="iou"), None),
        "T1-nocmc": (dict(motion="image", cmc=False, assoc="iou"), None),
        # T2: time-scale-aware fusion — image stage for continuing tracks, pitch-plane
        # physics stage for lost tracks; calibration = offline two-filter smoother
        "T2": (P, "smoothL"),
        "T2-allstage-c1": ({**P, "pitch_stage": "all"}, "c1"),     # first (naive) design
        "T2-allstage": ({**P, "pitch_stage": "all"}, "smoothL"),
        "T2-c1L": (P, "c1L"),                                        # per-frame line sensor, no filter
        "T2-fusedL": (P, "fusedL"),                                  # online filter
        "T2-pointsensor": (P, "smooth"),                             # smoother on the old point sensor
        "T2-isoR": ({**P, "het_R": False}, "smoothL"),
        "T2-novmax": ({**P, "v_max": None}, "smoothL"),
        "T3": ({**P, "assoc": "fusion", "appearance": True, "app_lr": app_lr}, "smoothL"),
        "T4": ({**P, "assoc": "fusion", "appearance": True, "app_lr": app_lr, "stitch": True}, "smoothL"),
        "T4-noapp": ({**P, "stitch": True}, "smoothL"),
    }
    kw, calib = V[name]
    return TrackerConfig(fps=fps, **kw), calib


def run_repo_bytetrack(seq, dets, fixed=False):
    """T0: the production tracker as found in the repo (frozen copy), only de-duplicating
    its output. T0-fixed: pipeline/bytetrack.py after the engineering fix."""
    if fixed:
        from pipeline.bytetrack import BYTETracker
    else:
        from baselines.bytetrack_repo_orig import BYTETracker
    trk = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=int(seq.fps))
    rows = []
    for f in range(1, seq.length + 1):
        d = dets[f]
        d = d[d[:, 5] != 3] if len(d) else d
        inp = [dict(object_id=int(r[5]), pixel_x=r[0] + r[2] / 2, pixel_y=r[1] + r[3] / 2,
                    width=r[2], height=r[3], confidence=r[4]) for r in d]
        seen = set()   # the original tracker emits duplicated tracks; keep one per id
        for o in trk.update(inp):
            if o["track_id"] in seen:
                continue
            seen.add(o["track_id"])
            rows.append((f, o["track_id"], o["pixel_x"] - o["width"] / 2, o["pixel_y"] - o["height"] / 2,
                         o["width"], o["height"], o["confidence"]))
    return rows


# Published trackers (boxmot 26.0.0), same detections/roles/evaluation as ours. For fairness the
# track-loss horizon is matched to ours (75 frames = 3 s) and min_hits=1 (as ours); everything
# else is the library default. ReID off everywhere (ours uses a non-learned colour histogram).
BOXMOT = {
    "X-bytetrack": ("ByteTrack", dict(min_hits=1, max_age=75, max_lost_frames=75)),
    "X-ocsort": ("OcSort", dict(min_hits=1, max_age=75, use_cmc=False)),
    "X-ocsort-cmc": ("OcSort", dict(min_hits=1, max_age=75, use_cmc=True)),
    "X-botsort": ("BotSort", dict(min_hits=1, max_age=75, max_lost_frames=75, use_cmc=True,
                                  use_embeddings=False)),
}


def run_boxmot(seq, dets, variant, overrides=None):
    import dataclasses
    import importlib
    import boxmot
    cls_name, kw = BOXMOT[variant]
    kw = {**kw, **(overrides or {})}                    # tuned thresholds (scripts/tune_tracking.py)
    mod = {"ByteTrack": "bytetrack", "OcSort": "ocsort", "BotSort": "botsort"}[cls_name]
    Cfg = getattr(importlib.import_module(f"boxmot.trackers.{mod}.config"), f"{cls_name}Config")
    trk = getattr(boxmot, cls_name)(dataclasses.replace(Cfg(), **kw))
    need_img = kw.get("use_cmc", False)
    rows = []
    for f in range(1, seq.length + 1):
        d = dets[f]
        d = d[d[:, 5] != 3] if len(d) else d                         # persons only (ball has its own tracker)
        arr = np.zeros((len(d), 6), np.float32)
        if len(d):
            arr[:, 0:2] = d[:, 0:2]; arr[:, 2:4] = d[:, 0:2] + d[:, 2:4]; arr[:, 4] = d[:, 4]; arr[:, 5] = d[:, 5]
        img = cv2.imread(str(seq.img_path(f))) if need_img else np.zeros((seq.height, seq.width, 3), np.uint8)
        out = trk.update(arr, img)
        out = out[0] if isinstance(out, tuple) else out
        out = np.asarray(out)
        for r in out.reshape(-1, out.shape[-1] if out.size else 8):
            x0, y0, x1, y1, tid, sc = r[:6]
            k = int(r[7]) if len(r) > 7 else -1
            if 0 <= k < len(d):              # report the matched detection box (same convention as ours)
                x0, y0, w, h = d[k, :4]; x1, y1 = x0 + w, y0 + h
            rows.append((f, int(tid), x0, y0, x1 - x0, y1 - y0, sc))
    return rows


def run_one(seq_name, split, variant, det_mode, out_root, app_lr_path, overrides=None):
    seq = load_sequence(SNMOT_ROOT / split / seq_name)
    out = out_root / f"{seq_name}.txt"
    dets = oracle_detections(seq) if det_mode == "oracle" else load_det_cache(seq, split, det_mode)
    if variant in BOXMOT:
        write_mot(out, run_boxmot(seq, dets, variant, overrides))
        return seq_name
    if variant in ("T0", "T0-fixed"):
        write_mot(out, run_repo_bytetrack(seq, dets, fixed=variant == "T0-fixed"))
        return seq_name
    cache = np.load(ROOT / "cache" / "snmot" / split / f"{seq_name}.npz")
    kp, M = cache["kp"], cache["M"]
    cfg, calib_mode = variant_config(variant, seq.fps, app_lr_from_file(app_lr_path))
    for k, val in (overrides or {}).items():
        setattr(cfg, k, val)
    trk = MultiTracker(cfg)
    calibs = sequence_calibrations(calib_mode or "none", kp, M, seq.width, seq.height)
    for f in range(1, seq.length + 1):
        H, s = calibs[f]
        img = cv2.imread(str(seq.img_path(f))) if cfg.appearance else None
        trk.step(f, dets[f], M=M[f] if cfg.cmc else None, H=H, sigma_cal=min(s, 5.0), img=img)
    write_mot(out, trk.results())
    return seq_name


def load_det_cache(seq, split, det_mode):
    from fcs.tracking.data import detections_by_frame
    p = ROOT / "cache" / "dets" / det_mode / split / f"{seq.name}.txt"
    return detections_by_frame(np.loadtxt(p, delimiter=",", ndmin=2), seq.length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", nargs="+", default=["T0", "T1", "T2"])
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--det", default="oracle")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--app-lr", default=str(ROOT / "cache" / "app_lr.json"))
    ap.add_argument("--set", nargs="*", default=[], help="TrackerConfig overrides k=v (our trackers only)")
    ap.add_argument("--tag", default="", help="suffix for output folders, e.g. -s05")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    overrides = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in a.set)}
    seqs = a.seqs or sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
    from fcs.tracking.evaluate import evaluate
    dirs, summary = {}, {}
    prog = Progress(len(seqs) * len(a.variants), f"tracking {a.split}/{a.det}", "seq")
    for v in a.variants:
        out_root = ROOT / "outputs" / "tracking" / a.split / a.det / (v + a.tag)
        out_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(partial(run_one, split=a.split, variant=v, det_mode=a.det,
                                                 out_root=out_root, app_lr_path=Path(a.app_lr),
                                                 overrides=overrides), seqs):
                prog.update(1, desc=f"tracking {a.split}/{a.det} variant {v}")
        print(f"{v}: ran {len(seqs)} seqs in {time.time() - t0:.0f}s", flush=True)
        dirs[v + a.tag] = out_root
    prog.close()
    if a.no_eval:
        return
    res = evaluate(dirs, split=a.split, seqs=seqs, roles=[0, 1, 2, 4],
                   work=ROOT / "cache" / f"eval_{a.split}_{a.det}")
    for v, r in res.items():
        print(f"{v:10s} " + "  ".join(f"{k} {val:6.2f}" if isinstance(val, float) else f"{k} {val}" for k, val in r.items() if not k.startswith("_")))
        summary[v] = r
    sp = ROOT / "outputs" / "tracking" / a.split / a.det / f"summary_{'_'.join(a.variants)}{a.tag}.json"
    sp.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
