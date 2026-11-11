"""Re-implementation of the SoccerNet-Calibration camera metric (evaluate_camera.py logic),
written from reading the official source because executing it was blocked here.
Must be validated against the official script before numbers are reported.

Per image (at 960x540):
  * project every pitch marking with the predicted camera (points sampled along it);
    a class is "predicted" if any projected point falls inside the image;
  * FP class: predicted but not annotated (+1 to FP); FN class: annotated, not predicted (+1 FN);
  * common class: TP if *all* annotated points are within `thr` px of the projected
    polyline, else +1 FP;
  * accuracy = TP / (TP + FP + FN); the better of {labels, mirrored labels} is kept.
Dataset: completeness = fraction of images with a prediction; JaC@thr = mean accuracy over
predicted images; final score = completeness × JaC@5.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .pitch import ALL_CLASSES, CIRCLES, LINES, mirror_name
from ..geometry.projection import Camera

W_EVAL, H_EVAL = 960, 540


def sample_marking(name: str, step_m: float = 0.5) -> np.ndarray:
    if name in LINES:
        a, b = LINES[name]
        n = max(2, int(np.linalg.norm(b - a) / step_m) + 1)
        t = np.linspace(0, 1, n)[:, None]
        return a + t * (b - a)
    c, r, (t0, t1) = CIRCLES[name]
    n = max(8, int(r * (t1 - t0) / step_m) + 1)
    t = np.linspace(t0, t1, n)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t), np.zeros(n)]


_SAMPLES = {n: sample_marking(n) for n in ALL_CLASSES}


def project_markings(cam: Camera, w: int = W_EVAL, h: int = H_EVAL) -> Dict[str, np.ndarray]:
    out = {}
    for n, X in _SAMPLES.items():
        front = cam.depth(X) > 1e-5
        if not front.any():
            continue
        u = cam.project(X[front])
        inside = (u[:, 0] >= 0) & (u[:, 0] < w) & (u[:, 1] >= 0) & (u[:, 1] < h)
        if inside.any():
            # keep the in-image run(s) plus one neighbour on each side (approximates the
            # official border-intersection handling)
            idx = np.where(inside)[0]
            lo, hi = max(0, idx.min() - 1), min(len(u) - 1, idx.max() + 1)
            out[n] = u[lo:hi + 1]
    return out


def _dist_to_polyline(p: np.ndarray, poly: np.ndarray) -> float:
    if len(poly) == 1:
        return float(np.linalg.norm(p - poly[0]))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


def image_accuracy(pred: Dict[str, np.ndarray], gt: Dict[str, np.ndarray], thr: float) -> float:
    tp = fp = fn = 0
    for c in set(pred) - set(gt):
        fp += 1
    for c in set(gt) - set(pred):
        fn += 1
    for c in set(pred) & set(gt):
        if all(_dist_to_polyline(q, pred[c]) < thr for q in gt[c]):
            tp += 1
        else:
            fp += 1
    tot = tp + fp + fn
    return tp / tot if tot else 0.0


def evaluate_image(cam: Optional[Camera], gt_norm: Dict[str, list], thr: float) -> Optional[float]:
    """gt_norm: SN annotation dict (normalised x, y). Returns accuracy or None if no camera."""
    if cam is None:
        return None
    gt = {k: np.array([[q["x"] * W_EVAL, q["y"] * H_EVAL] for q in v]) for k, v in gt_norm.items()
          if k in set(ALL_CLASSES) and v}
    pred = project_markings(cam)
    gt_m = {mirror_name(k): v for k, v in gt.items()}
    return max(image_accuracy(pred, gt, thr), image_accuracy(pred, gt_m, thr))


def summarise(accs: list, thr_list=(5, 10, 20)) -> dict:
    """accs: list of dicts {thr: accuracy or None} per image."""
    n = len(accs)
    have = [a for a in accs if a[thr_list[0]] is not None]
    res = {"n_images": n, "completeness": len(have) / max(1, n)}
    for t in thr_list:
        res[f"JaC@{t}"] = float(np.mean([a[t] for a in have])) if have else 0.0
    res["final_score"] = res["completeness"] * res["JaC@5"]
    return res
