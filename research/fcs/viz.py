"""Helpers for notebooks/comparison.ipynb: load result files, find illustrative examples, draw them.
Nothing here computes a reported number; numbers come from the JSON files written by the scripts."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

NICE = {"T0": "Thesis tracker (T0)", "T0-fixed": "Thesis tracker, bugs fixed", "T1": "Kalman + CMC (T1)",
        "T2": "+ pitch-plane physics (T2)", "T3": "+ appearance LLR (T3)", "T4": "Ours, full (T4)",
        "X-bytetrack": "ByteTrack", "X-ocsort": "OC-SORT", "X-ocsort-cmc": "OC-SORT + CMC",
        "X-botsort": "BoT-SORT"}


def nice(v: str) -> str:
    base, tuned = (v[:-6], True) if v.endswith("-tuned") else (v, False)
    return NICE.get(base, base) + (" — tuned" if tuned else "")


# ───────────────────────────── tracking tables ─────────────────────────────
def tracking_table(split: str, det: str, variants: List[str]) -> pd.DataFrame:
    rows = {}
    for f in sorted((OUT / "tracking" / split / det).glob("summary_*.json"), key=lambda p: p.stat().st_mtime):  # newest wins
        for k, v in json.loads(f.read_text()).items():
            rows[k] = {m: v[m] for m in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")}
    df = pd.DataFrame([dict(tracker=nice(v), key=v, **rows[v]) for v in variants if v in rows])
    return df.set_index("tracker")


def significance_table(split: str, det: str, ours: str = "T4") -> Optional[pd.DataFrame]:
    p = OUT / "tracking" / split / det / f"significance_{ours}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["pairs"]
    rows = []
    for b, ms in d.items():
        for m, r in ms.items():
            rows.append(dict(comparison=f"{nice(ours)} vs {nice(b)}", metric=m,
                             mean_diff=round(r["mean"], 2), ci95=f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]",
                             wilcoxon_p=f"{r['wilcoxon_p']:.1e}", wins_ties_losses=f"{r['wins']}/{r['ties']}/{r['losses']}"))
    return pd.DataFrame(rows).set_index(["comparison", "metric"])


# ───────────────────────────── ID-switch examples ──────────────────────────
def _read(path: Path) -> np.ndarray:
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a if a.size else np.zeros((0, 10))


def _iou(a, B):
    x0 = np.maximum(a[0], B[:, 0]); y0 = np.maximum(a[1], B[:, 1])
    x1 = np.minimum(a[0] + a[2], B[:, 0] + B[:, 2]); y1 = np.minimum(a[1] + a[3], B[:, 1] + B[:, 3])
    i = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return i / (a[2] * a[3] + B[:, 2] * B[:, 3] - i + 1e-9)


def _assignments(gt: np.ndarray, pr: np.ndarray, frames) -> Dict[int, Dict[int, int]]:
    """gt id -> {frame: predicted id} via best IoU >= 0.5 (illustration only)."""
    out: Dict[int, Dict[int, int]] = {}
    for f in frames:
        G, P = gt[gt[:, 0] == f], pr[pr[:, 0] == f]
        if not len(P):
            continue
        for g in G:
            io = _iou(g[2:6], P[:, 2:6]); k = int(np.argmax(io))
            if io[k] >= 0.5:
                out.setdefault(int(g[1]), {})[int(f)] = int(P[k, 1])
    return out


def find_switch_examples(split: str, det: str, ours: str, base: str, seqs: List[str], max_n: int = 3):
    """Moments where `base` changes the identity of a person while `ours` keeps it."""
    from .tracking.data import SNMOT_ROOT, load_sequence
    found = []
    for s in seqs:
        seq = load_sequence(SNMOT_ROOT / split / s)
        gt = seq.gt(); gt = gt[[seq.role_of(i) != 3 for i in gt[:, 1]]]
        A = {v: _assignments(gt, _read(OUT / "tracking" / split / det / v / f"{s}.txt"), range(1, seq.length + 1))
             for v in (ours, base)}
        for gid, seq_b in A[base].items():
            fr = sorted(seq_b)
            for f0, f1 in zip(fr, fr[1:]):
                if f1 - f0 > 3 or seq_b[f0] == seq_b[f1]:
                    continue
                o = A[ours].get(gid, {})
                if f0 in o and f1 in o and o[f0] == o[f1]:
                    found.append(dict(seq=s, gt_id=gid, f0=f0, f1=f1)); break
            if len(found) >= max_n:
                return found
    return found


def _colour(i: int):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_tracks(split: str, det: str, variant: str, seq_name: str, frame: int, focus_gt: Optional[int] = None,
                crop_pad: int = 220) -> np.ndarray:
    """RGB crop around the focus person with every track box + id drawn (colour = id)."""
    from .tracking.data import SNMOT_ROOT, load_sequence
    seq = load_sequence(SNMOT_ROOT / split / seq_name)
    img = cv2.imread(str(seq.img_path(frame)))
    pr = _read(OUT / "tracking" / split / det / variant / f"{seq_name}.txt"); pr = pr[pr[:, 0] == frame]
    for r in pr:
        x, y, w, h = r[2:6].astype(int); c = _colour(r[1])
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        cv2.putText(img, str(int(r[1])), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    if focus_gt is not None:
        gt = seq.gt(); g = gt[(gt[:, 0] == frame) & (gt[:, 1] == focus_gt)]
        if len(g):
            x, y, w, h = g[0, 2:6].astype(int)
            cv2.rectangle(img, (x - 4, y - 4), (x + w + 4, y + h + 4), (255, 255, 255), 1)
            cx, cy = x + w // 2, y + h // 2
            x0, y0 = max(0, cx - crop_pad), max(0, cy - crop_pad)
            img = img[y0:cy + crop_pad, x0:cx + crop_pad]
    return img[..., ::-1]


# ───────────────────────────── calibration overlays ────────────────────────
def calib_overlay(tag: str, zip_path: Path, index: int):
    """Project the pitch model through the camera solved from PitchGeo predictions; draw on the image.
    Returns (RGB image 960x540, accuracy@5px or None)."""
    import torch
    from .calib.sn_metric import H_EVAL, W_EVAL, evaluate_image, project_markings
    from .calib.solve import from_network, solve_camera
    keys = ("presence", "log_sigma", "n", "rho", "centre", "axes", "orient")
    d = np.load(OUT / "pitchgeo" / "eval" / f"{tag}_raw.npz")
    o = {k: torch.from_numpy(d[k][index:index + 1]) for k in keys}
    sol = solve_camera(from_network(o, 0, W_EVAL, H_EVAL), priors=True, distortion=True)
    name = str(d["names"][index])
    z = zipfile.ZipFile(zip_path)
    img = cv2.imdecode(np.frombuffer(z.read(name[:-5] + ".jpg"), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (W_EVAL, H_EVAL))
    ann = json.loads(z.read(name))
    for pts in ann.values():                              # ground truth: white dots
        for q in pts:
            cv2.circle(img, (int(q["x"] * W_EVAL), int(q["y"] * H_EVAL)), 3, (255, 255, 255), -1)
    acc = None
    if sol.camera is not None:
        for poly in project_markings(sol.camera).values():   # prediction: magenta polylines
            if len(poly) >= 2:
                cv2.polylines(img, [poly.astype(np.int32)], False, (255, 0, 255), 2)
        acc = evaluate_image(sol.camera, ann, 5)
    return img[..., ::-1], acc


# ───────────────────────────── ball trajectories ───────────────────────────
def ball_series(split: str, det: str, variant: str, seq_name: str) -> pd.Series:
    p = OUT / "ball" / split / det / variant / f"{seq_name}.txt"
    a = _read(p) if p.exists() else np.zeros((0, 10))
    return pd.Series(a[:, 2] + a[:, 4] / 2, index=a[:, 0].astype(int)) if len(a) else pd.Series(dtype=float)


def ball_gt_series(split: str, seq_name: str) -> pd.Series:
    from .tracking.data import SNMOT_ROOT, load_sequence
    seq = load_sequence(SNMOT_ROOT / split / seq_name)
    gt = seq.gt(); gt = gt[[seq.role_of(i) == 3 for i in gt[:, 1]]]
    return pd.Series(gt[:, 2] + gt[:, 4] / 2, index=gt[:, 0].astype(int))


# ───────────────────────────── commentary samples ──────────────────────────
def caption_samples(files: Dict[str, Path], n: int = 6, seed: int = 0) -> pd.DataFrame:
    tabs = {}
    for k, p in files.items():
        if p.exists():
            tabs[k] = {(r["game"], r["half"], r["t"]): r for r in map(json.loads, open(p))}
    common = sorted(set.intersection(*[set(t) for t in tabs.values()]))
    pick = np.random.default_rng(seed).choice(len(common), min(n, len(common)), replace=False)
    first = next(iter(tabs.values()))
    rows = []
    for i in pick:
        key = common[i]
        rows.append({"game state (input, truncated)": first[key]["state"][:160] + "…",
                     "reference": first[key]["ref"], **{k: t[key]["pred"] for k, t in tabs.items()}})
    return pd.DataFrame(rows)
