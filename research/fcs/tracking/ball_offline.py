"""Offline (whole-clip) ball trackers B4–B6.

B4  Global path search. One ball → find the minimum-cost path through all candidates (score ≥ s_min)
    with an explicit "ball not visible" state. Cost of a path =
      Σ node costs  −a·log(score / s_ref)               (detector evidence; reward above s_ref)
    + Σ motion costs between consecutive path nodes, k frames apart:
          ½ (d / σ_k)²,  d = |p_j − p_i| in image coordinates (optionally camera-motion-compensated;
          off by default — the broadcast camera follows the ball, so CMC *doubles* the ball's
          frame-to-frame step, 6 → 12 px median on SNMOT-060), σ_k = σ0 + σ1·k
    + β_gap per bridged (undetected) frame,  β_abs per "not visible" frame,  γ to (re-)enter.
    Exact by dynamic programming over a DAG (transitions up to G frames).  Bridged gaps ≤ L
    frames are filled by motion-compensated linear interpolation.
B5  + geometric size prior: expected ball diameter at the candidate's location from the image→pitch
    homography (local metric scale × 0.22 m); node cost += ½ ((log w − log w_exp)/σ_s)².
B6  + per-video online appearance model: from the B5 path, confident nodes = positives, rejected
    candidates far from the path = negatives; fit a logistic regression on patch colour features
    for THIS clip, add λ·(its logit) to every candidate's evidence, search again (EM-like, n_iter).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ..geometry.camera import apply_h, h_jacobian

BALL_D = 0.22


@dataclass
class OfflineBallConfig:
    variant: str = "B4"
    s_min: float = 0.05
    a: float = 1.0
    s_ref: float = 0.2        # detections scoring above s_ref lower the path cost (reward), below raise it
    cmc: bool = False         # the broadcast camera FOLLOWS the ball: raw image coords are the better motion frame
    sigma0: float = 4.0       # px
    sigma1: float = 2.5       # px per frame of separation
    beta_gap: float = 0.25    # per bridged frame
    beta_abs: float = 1.2     # per "not visible" frame (≈ −log 0.3)
    gamma: float = 1.5        # (re-)enter cost
    G: int = 25               # max frames bridged by one transition
    L: int = 25               # max gap filled by interpolation
    sigma_size: float = 0.45  # log-size std (B5)
    size_ratio: float = 2.14  # annotated box / geometric ball diameter, median on the tuning split (blur, loose boxes)
    w_size: float = 1.0
    lam_app: float = 1.0      # B6 weight
    n_iter: int = 2           # B6 iterations
    pos_min_score: float = 0.4
    neg_min_dist: float = 40.0


# ─────────────────────────────── helpers ───────────────────────────────
def _ok(M):
    return M is not None and np.all(np.isfinite(M)) and abs(np.linalg.det(M)) > 1e-8


class _Warp:
    """Composite inter-frame homographies M_{f→g} = M[g] ··· M[f+1] (cached)."""

    def __init__(self, M):
        self.M = M; self.cache: Dict[tuple, np.ndarray] = {}

    def __call__(self, f: int, g: int) -> np.ndarray:
        key = (f, g)
        if key not in self.cache:
            if g == f:
                W = np.eye(3)
            else:
                W = self(f, g - 1)
                Mg = self.M[g] if g < len(self.M) and _ok(self.M[g]) else np.eye(3)
                W = Mg @ W
            self.cache[key] = W
        return self.cache[key]


def expected_diameter(H: Optional[np.ndarray], c: np.ndarray) -> Optional[float]:
    """Ball diameter (px) expected at image point c from the image→pitch homography H."""
    if H is None or not np.all(np.isfinite(H)):
        return None
    try:
        J = h_jacobian(np.linalg.inv(H), apply_h(H, c[None])[0])   # pitch → image, px per metre
    except np.linalg.LinAlgError:
        return None
    s = np.sqrt(abs(np.linalg.det(J)))
    return float(BALL_D * s) if np.isfinite(s) and s > 0 else None


def patch_features(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """HSV histogram of the (1.3× enlarged) box + its centre-ring contrast — cheap appearance code."""
    x, y, w, h = box
    cx, cy, r = x + w / 2, y + h / 2, 0.65 * max(w, h, 4)
    x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
    p = img[y0:y1, x0:x1]
    if p.size == 0:
        return np.zeros(8 * 4 * 4 + 2)
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist /= hist.sum() + 1e-9
    v = hsv[..., 2].astype(np.float32)
    return np.r_[np.sqrt(hist), v.mean() / 255, v.std() / 255]


# ─────────────────────────────── path search ───────────────────────────
def _search(cands: List[np.ndarray], node_cost: List[np.ndarray], warp: _Warp, cfg: OfflineBallConfig):
    """cands[f]: (n_f, 2) centres; node_cost[f]: (n_f,). Returns list of (frame, idx) on the best path."""
    T = len(cands) - 1
    D: List[np.ndarray] = [np.zeros(0) for _ in range(T + 1)]
    back: List[list] = [[] for _ in range(T + 1)]
    A = np.zeros(T + 1)                     # best cost with "not visible" at frame f
    A_back = [None] * (T + 1)
    for f in range(1, T + 1):
        # not-visible state
        best_prev, arg = A[f - 1], ("A", f - 1)
        if len(D[f - 1]):
            i = int(np.argmin(D[f - 1]))
            if D[f - 1][i] < best_prev:
                best_prev, arg = D[f - 1][i], ("D", f - 1, i)
        A[f] = best_prev + cfg.beta_abs; A_back[f] = arg
        n = len(cands[f])
        D[f] = np.full(n, np.inf); back[f] = [None] * n
        if not n:
            continue
        enter = A[f - 1] + cfg.gamma
        preds = []                                                    # predecessors carried to frame f
        for k in range(1, min(cfg.G, f - 1) + 1):
            g = f - k
            if len(cands[g]):
                preds.append((g, k, apply_h(warp(g, f), cands[g])))
        for j in range(n):
            bc, bb = enter, ("A", f - 1)
            for g, k, Wp in preds:
                d = np.linalg.norm(Wp - cands[f][j], axis=1)
                sig = cfg.sigma0 + cfg.sigma1 * k
                c = D[g] + 0.5 * (d / sig) ** 2 + cfg.beta_gap * (k - 1)
                i = int(np.argmin(c))
                if c[i] < bc:
                    bc, bb = c[i], ("D", g, i)
            D[f][j] = bc + node_cost[f][j]; back[f][j] = bb
    # termination: best over all end states
    best, state = A[T], ("A", T)
    if len(D[T]) and D[T].min() < best:
        state = ("D", T, int(np.argmin(D[T])))
    # paths may also end early (then "not visible" to the end) — handled by A's recursion
    path = []
    while state is not None:
        if state[0] == "D":
            _, f, j = state
            path.append((f, j)); state = back[f][j]
        else:
            _, f = state
            state = A_back[f] if f > 0 else None
            if state is not None and state[0] == "A" and state[1] == 0:
                break
    return path[::-1]


def track_offline(seq_len: int, balls: List[np.ndarray], M, cal, cfg: OfflineBallConfig,
                  read_img: Optional[Callable[[int], np.ndarray]] = None) -> List[tuple]:
    """balls[f]: (n, 6) [x, y, w, h, score, cls]. Returns rows (frame, x, y, w, h, score)."""
    warp = _Warp(M if cfg.cmc else [None] * len(M))
    B = [np.zeros((0, 6))] + [b[b[:, 4] >= cfg.s_min] if len(b) else np.zeros((0, 6)) for b in balls[1:seq_len + 1]]
    C = [b[:, :2] + b[:, 2:4] / 2 for b in B]
    evid = [np.log(np.clip(b[:, 4], 1e-4, 1)) - np.log(cfg.s_ref) for b in B]   # log-score evidence vs reference

    def node_costs(extra=None):
        out = []
        for f, b in enumerate(B):
            c = -cfg.a * evid[f]
            if extra is not None:
                c = c - extra[f]
            if cfg.variant in ("B5", "B6") and len(b):
                for j in range(len(b)):
                    de = expected_diameter(cal[f][0] if f < len(cal) else None, C[f][j])
                    if de:
                        dw = np.log(max(np.sqrt(b[j, 2] * b[j, 3]), 1.0)) - np.log(cfg.size_ratio * de)
                        c[j] += cfg.w_size * 0.5 * (dw / cfg.sigma_size) ** 2
            out.append(c)
        return out

    nc = node_costs()
    path = _search(C, nc, warp, cfg)
    if cfg.variant == "B6" and read_img is not None:
        from sklearn.linear_model import LogisticRegression
        feats: Dict[tuple, np.ndarray] = {}
        for f in range(1, seq_len + 1):
            if len(B[f]):
                img = read_img(f)
                for j in range(len(B[f])):
                    feats[(f, j)] = patch_features(img, B[f][j, :4])
        for _ in range(cfg.n_iter):
            on = {f: j for f, j in path}
            pos = [feats[(f, j)] for f, j in path if B[f][j, 4] >= cfg.pos_min_score]
            neg = []
            for f in range(1, seq_len + 1):
                if f in on:
                    for j in range(len(B[f])):
                        if j != on[f] and np.linalg.norm(C[f][j] - C[f][on[f]]) > cfg.neg_min_dist:
                            neg.append(feats[(f, j)])
            if len(pos) < 10 or len(neg) < 10:
                break
            X = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(X, y)
            extra = [np.zeros(len(b)) for b in B]
            for (f, j), x in feats.items():
                extra[f][j] = cfg.lam_app * float(clf.decision_function(x[None])[0])
            path = _search(C, node_costs(extra), warp, cfg)

    # outputs: path nodes + motion-compensated interpolation of bridged gaps ≤ L
    rows = []
    for idx, (f, j) in enumerate(path):
        b = B[f][j]
        rows.append((f, *b[:4], float(b[4])))
        if idx + 1 < len(path):
            g, i = path[idx + 1]
            if 1 < g - f <= cfg.L + 1:
                bg = B[g][i]
                for q in range(f + 1, g):
                    w = (q - f) / (g - f)
                    p0 = apply_h(warp(f, q), C[f][j][None])[0]
                    p1 = apply_h(np.linalg.inv(warp(q, g)), C[g][i][None])[0]
                    c = (1 - w) * p0 + w * p1
                    sz = (1 - w) * b[2:4] + w * bg[2:4]
                    rows.append((q, c[0] - sz[0] / 2, c[1] - sz[1] / 2, sz[0], sz[1], 0.3))
    return rows
