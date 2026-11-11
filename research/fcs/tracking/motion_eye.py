"""Training-free small-target motion eye (ESTMD-inspired: dragonfly small-target-motion-detector neurons).

Causal: at frame t it uses frames t-2, t-1, t only.
  1. grey, half resolution; previous frames warped into frame t with the ego-motion homographies (the background is
     aligned, so anything that still changes moves relative to the pitch);
  2. three-frame difference  D = min(|I_t − I_{t-1}|, |I_t − I_{t-2}|)  — high only where something is NOW
     (a single difference also fires at where the ball WAS);
  3. centre-surround (difference of Gaussians at ball scale) — lateral inhibition favours small blobs over edges
     and large moving bodies;
  4. players (detector boxes, dilated) and the top band of the frame (crowd/boards) are suppressed;
  5. local maxima → candidates, score = robust z-score mapped to (0, 1).
When the tracker has a prediction, the search is restricted to a window of radius k·σ_pred (cheaper, fewer distractors).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

SCALE = 0.5


class MotionEye:
    def __init__(self, seq, M, dets, top_frac: float = 0.0, k_sigma: float = 4.0, min_r: float = 60.0,
                 z0: float = 4.0, zk: float = 1.5, max_cands: int = 5):
        self.seq, self.M, self.dets = seq, M, dets
        self.top_frac, self.k_sigma, self.min_r, self.z0, self.zk, self.max_cands = top_frac, k_sigma, min_r, z0, zk, max_cands
        self.buf: Dict[int, np.ndarray] = {}
        S = np.diag([SCALE, SCALE, 1.0]); self.S, self.Si = S, np.linalg.inv(S)

    def _grey(self, f):
        if f not in self.buf:
            img = cv2.imread(str(self.seq.img_path(f)))
            g = cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            self.buf[f] = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
            for k in [k for k in self.buf if k < f - 2]:
                del self.buf[k]
        return self.buf[f]

    def _Mh(self, f):
        """Half-resolution homography frame f-1 -> f."""
        M = self.M[f] if f < len(self.M) and np.all(np.isfinite(self.M[f])) else np.eye(3)
        return self.S @ M @ self.Si

    def __call__(self, f, xp=None, Pp=None, trk=None) -> List[tuple]:
        if f < 3:
            self._grey(f); return []
        I0, I1, I2 = self._grey(f), self._grey(f - 1), self._grey(f - 2)
        h, w = I0.shape
        M1 = self._Mh(f); M2 = M1 @ self._Mh(f - 1)
        W1 = cv2.warpPerspective(I1, M1, (w, h), borderMode=cv2.BORDER_REPLICATE)
        W2 = cv2.warpPerspective(I2, M2, (w, h), borderMode=cv2.BORDER_REPLICATE)
        D = np.minimum(np.abs(I0 - W1), np.abs(I0 - W2))
        R = cv2.GaussianBlur(D, (0, 0), 1.5) - cv2.GaussianBlur(D, (0, 0), 5.0)      # centre-surround
        mask = np.ones_like(R, bool)
        mask[: int(self.top_frac * h)] = False
        d = self.dets[f]
        if len(d):
            for x, y, bw, bh, s, c in d[:, :6]:
                if c != 3 and s >= 0.3:                                           # players / referees
                    x0, y0 = int(SCALE * (x - 0.1 * bw)), int(SCALE * y)
                    x1, y1 = int(SCALE * (x + 1.1 * bw)), int(SCALE * (y + 0.85 * bh))   # keep the feet band
                    mask[max(0, y0):y1, max(0, x0):x1] = False
        if xp is not None and Pp is not None:
            r = max(self.min_r, self.k_sigma * float(np.sqrt(np.max(np.linalg.eigvalsh(Pp[:2, :2]))))) * SCALE
            cx, cy = xp[0] * SCALE, xp[1] * SCALE
            win = np.zeros_like(mask); win[max(0, int(cy - r)):int(cy + r), max(0, int(cx - r)):int(cx + r)] = True
            mask &= win
        Rm = np.where(mask, R, 0.0)
        vals = R[mask]
        if vals.size < 50:
            return []
        med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-3
        peaks = (Rm == cv2.dilate(Rm, np.ones((7, 7), np.uint8))) & (Rm > med + self.z0 * mad)
        ys, xs = np.nonzero(peaks)
        if not len(xs):
            return []
        z = (Rm[ys, xs] - med) / mad
        order = np.argsort(-z)[: self.max_cands]
        out = []
        for k in order:
            score = float(1 / (1 + np.exp(-(z[k] - self.z0 - 2.0) / self.zk)))
            out.append((xs[k] / SCALE, ys[k] / SCALE, 14.0, 14.0, score))
        return out
