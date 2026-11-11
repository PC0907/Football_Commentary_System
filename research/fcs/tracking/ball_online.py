"""Causal (real-time) multi-eye ball tracker.  At frame t it uses only frames ≤ t.

State: image-plane constant-velocity Kalman filter [x, y, vx, vy] — no camera-motion compensation (the broadcast
camera follows the ball, so compensating its motion doubles the ball's frame-to-frame step).  Each frame, candidates arrive from several "eyes":
  periph   full-frame detector (every frame)
  sahi     sliced native-resolution detector (consulted 'always' or only 'lost' — the saccade policy)
  motion   camera-motion-compensated small-target motion detector (ESTMD-like, training-free; fcs/tracking/motion_eye.py)
  fovea    learned temporal network on a crop around the prediction (fcs/ball/fovea.py)
Candidate selection maximises   Σ evidence  a_e·log(s / s_ref_e)  +  log N(z; Hx̂, S)   over gated candidates
(duplicates from different eyes within `merge_px` are merged; their evidences add — independent sensors).
Per-video online learning (causal):
  * adaptive process noise: Q ← λ_t Q with λ_t an EMA of the normalised innovation squared / dof (Sage–Husa-style
    covariance matching) — the filter learns how erratically THIS video's ball moves;
  * online score calibration: per-eye EMA of accepted-candidate scores sets that eye's s_ref (optional).
Track management: coast up to `coast` frames on prediction; after `lost_after` misses the track is LOST and the
best candidate with evidence ≥ τ_acq (any eye) re-acquires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class OnlineBallConfig:
    eyes: tuple = ("periph",)
    sahi_policy: str = "lost"         # 'lost' | 'always' | 'never'
    s_ref: Dict[str, float] = field(default_factory=lambda: dict(periph=0.2, sahi=0.3, motion=0.5, fovea=0.3))
    a: Dict[str, float] = field(default_factory=lambda: dict(periph=1.0, sahi=1.0, motion=0.5, fovea=1.0))
    s_min: float = 0.05
    q: float = 6.0                    # px/frame² white-acceleration std (base)
    r0: float = 2.0; r_rel: float = 0.15   # measurement std = r0 + r_rel·size
    gate: float = 25.0                # chi² gate on the 2-D innovation (loose: the ball changes direction)
    tau: float = -2.5                 # min total score to accept a gated candidate
    tau_acq: float = 0.7              # min evidence to (re)acquire
    tau_jump: float = 0.9             # evidence of an UNGATED candidate that makes the tracker jump to it (saccade)
    jump_eyes: tuple = ("periph", "sahi", "fovea")   # appearance eyes only: a lone motion blob may not hijack the track
    secondary_eyes: tuple = ("motion",)  # consulted only when no primary (appearance) candidate passes the gate
    size_from_track: tuple = ("motion",) # these eyes give no reliable box: use the track's running size
    secondary_min_speed: float = 0.0  # px/frame: consult the secondary (motion) eye only for a fast-moving track
    coast: int = 4
    lost_after: int = 8
    merge_px: float = 8.0
    adapt_q: bool = True; adapt_alpha: float = 0.05; lam_min: float = 0.3; lam_max: float = 20.0
    calib_scores: bool = False
    blur_velocity: bool = True        # use the fovea's blur vector as a velocity measurement
    blur_min: float = 4.0             # px/frame: below this the streak axis is unreliable
    blur_sig_rel: float = 0.35; blur_sig0: float = 2.0


class OnlineBallTracker:
    def __init__(self, cfg: OnlineBallConfig):
        self.cfg = cfg
        self.x: Optional[np.ndarray] = None; self.P = None
        self.size = np.array([14.0, 14.0]); self.miss = 0; self.lam = 1.0
        self.score_ema: Dict[str, float] = {}

    @property
    def lost(self):
        return self.x is None or self.miss > self.cfg.lost_after

    def predict(self):
        """Prior for frame t (used by the fovea before step())."""
        if self.x is None:
            return None, None
        F = np.eye(4); F[0, 2] = F[1, 3] = 1.0
        q = self.cfg.q * np.sqrt(self.lam if self.cfg.adapt_q else 1.0)
        G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        return F @ self.x, F @ self.P @ F.T + (q ** 2) * (G @ G.T)

    def _evidence(self, eye, s):
        cfg = self.cfg
        ref = self.score_ema.get(eye, cfg.s_ref.get(eye, 0.3)) if cfg.calib_scores else cfg.s_ref.get(eye, 0.3)
        return cfg.a.get(eye, 1.0) * float(np.log(max(s, 1e-4) / ref))

    def _merge(self, cands: List[tuple]) -> List[dict]:
        """cands: (cx, cy, w, h, score, eye). Merge near-duplicates across eyes; evidences add."""
        out: List[dict] = []
        # appearance eyes first: a merged candidate keeps the geometry (centre, box) of its best APPEARANCE detection;
        # secondary eyes (motion) only add evidence to it
        for cand in sorted(cands, key=lambda c: (c[5] in self.cfg.secondary_eyes, -c[4])):
            cx, cy, w, h, s, eye = cand[:6]; blur = cand[6] if len(cand) > 6 else None
            for o in out:
                if np.hypot(o["c"][0] - cx, o["c"][1] - cy) <= self.cfg.merge_px and eye not in o["eyes"]:
                    o["ev"] += self._evidence(eye, s); o["eyes"].add(eye); o["blur"] = o.get("blur") or blur; break
            else:
                out.append(dict(c=np.array([cx, cy]), wh=np.array([w, h]), s=s, ev=self._evidence(eye, s), eyes={eye}, eye=eye, blur=blur))
        return out

    def step(self, frame: int, cands: List[tuple]) -> Optional[tuple]:
        cfg = self.cfg
        C = self._merge([c for c in cands if c[4] >= cfg.s_min])
        xp, Pp = self.predict()
        best = None
        prim = [c for c in C if not (c["eyes"] <= set(cfg.secondary_eyes))]
        sec = [c for c in C if c["eyes"] <= set(cfg.secondary_eyes)]
        if xp is None or float(np.hypot(*xp[2:])) < cfg.secondary_min_speed:
            sec = []
        for c in sec:
            if c["eyes"] <= set(cfg.size_from_track):
                c["wh"] = self.size.copy()
        if xp is not None and not self.lost:
            for is_sec, c in [(False, c) for c in prim] + [(True, c) for c in sec]:
                if is_sec and best is not None:
                    break                                   # a primary candidate was accepted: ignore the secondary eye
                rs = cfg.r0 + cfg.r_rel * float(np.mean(c["wh"]))
                S = Pp[:2, :2] + rs ** 2 * np.eye(2); y = c["c"] - xp[:2]
                d2 = float(y @ np.linalg.solve(S, y))
                if d2 > cfg.gate:
                    continue
                tot = c["ev"] - 0.5 * d2 - 0.5 * np.log(np.linalg.det(S) / (2 * np.pi * 25.0))
                if tot >= cfg.tau and (best is None or tot > best[0]):
                    best = (tot, c, S, y, rs, d2)
        if best is not None:
            _, c, S, y, rs, d2 = best
            K = Pp[:, :2] @ np.linalg.inv(S)
            self.x = xp + K @ y; self.P = (np.eye(4) - K[:, :2] @ np.eye(2, 4)) @ Pp
            if cfg.blur_velocity and c.get("blur") and c["blur"][0] >= cfg.blur_min:
                self._blur_update(c["blur"])
            if cfg.adapt_q:
                self.lam = float(np.clip((1 - cfg.adapt_alpha) * self.lam + cfg.adapt_alpha * d2 / 2.0, cfg.lam_min, cfg.lam_max))
            self._accept(c); return (frame, *(c["c"] - c["wh"] / 2), *c["wh"], float(c["s"]))
        # no gated candidate: a confident candidate elsewhere wins at once (kick / saccade); else (re)acquire or coast
        Cj = [c for c in C if c["eyes"] & set(cfg.jump_eyes)]
        if Cj and not (self.lost or xp is None):
            c = max(Cj, key=lambda c: c["ev"])
            if c["ev"] >= cfg.tau_jump:
                self.x = np.r_[c["c"], 0.0, 0.0]; self.P = np.diag([9.0, 9.0, 400.0, 400.0])
                self._accept(c); return (frame, *(c["c"] - c["wh"] / 2), *c["wh"], float(c["s"]))
        if self.lost or xp is None:
            if Cj:
                c = max(Cj, key=lambda c: c["ev"])
                if c["ev"] >= cfg.tau_acq:
                    self.x = np.r_[c["c"], 0.0, 0.0]; self.P = np.diag([9.0, 9.0, 100.0, 100.0]); self.miss = 0
                    self._accept(c); return (frame, *(c["c"] - c["wh"] / 2), *c["wh"], float(c["s"]))
            if xp is not None:
                self.x, self.P = xp, Pp; self.miss += 1
            return None
        self.x, self.P = xp, Pp; self.miss += 1
        if self.miss <= cfg.coast:
            return (frame, *(xp[:2] - self.size / 2), *self.size, 0.2)
        return None

    def _blur_update(self, blur):
        """Motion blur as a velocity measurement: |v| = streak length, axis = streak angle, sign from the filter."""
        L, th = blur; u = np.array([np.cos(th), np.sin(th)])
        v = L * u if u @ self.x[2:] >= 0 else -L * u
        Hv = np.zeros((2, 4)); Hv[0, 2] = Hv[1, 3] = 1.0
        sr = self.cfg.blur_sig0 + self.cfg.blur_sig_rel * L
        S = Hv @ self.P @ Hv.T + sr ** 2 * np.eye(2); K = self.P @ Hv.T @ np.linalg.inv(S)
        self.x = self.x + K @ (v - self.x[2:]); self.P = (np.eye(4) - K @ Hv) @ self.P

    def _accept(self, c):
        self.miss = 0; self.size = 0.7 * self.size + 0.3 * c["wh"]
        for e in c["eyes"]:
            self.score_ema[e] = 0.95 * self.score_ema.get(e, self.cfg.s_ref.get(e, 0.3)) + 0.05 * 0.5 * c["s"]


def run_online(length: int, eyes_static: Dict[str, List[np.ndarray]], cfg: OnlineBallConfig,
               eyes_online: Optional[Dict[str, Callable]] = None) -> List[tuple]:
    """eyes_static[name][f]: (n, ≥5) [x, y, w, h, score, ...] per frame (causal per-frame detectors).
    eyes_online[name](f, prior_mean, prior_cov, tracker) -> list of (cx, cy, w, h, score) (may use frames ≤ f)."""
    trk = OnlineBallTracker(cfg); rows = []
    for f in range(1, length + 1):
        cands = []
        for name in cfg.eyes:
            if name in eyes_static:
                if name == "sahi" and (cfg.sahi_policy == "never" or (cfg.sahi_policy == "lost" and not trk.lost)):
                    continue
                d = eyes_static[name][f]
                cands += [(r[0] + r[2] / 2, r[1] + r[3] / 2, r[2], r[3], r[4], name, None) for r in d]
            elif eyes_online and name in eyes_online:
                xp, Pp = trk.predict()
                cands += [(*c[:5], name, (c[5], c[6]) if len(c) > 6 else None) for c in eyes_online[name](f, xp, Pp, trk)]
        r = trk.step(f, cands)
        if r is not None:
            rows.append(r)
    return rows
