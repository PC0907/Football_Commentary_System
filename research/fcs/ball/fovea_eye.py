"""Online fovea eye: runs the fovea network on a native-resolution crop around the tracker's prediction.

Causal: frames t-2, t-1, t only.  Called by fcs/tracking/ball_online.run_online as eyes_online['fovea'](f, x̂, P̂, trk).
When the tracker has no prediction (not yet acquired) the eye stays closed — the periphery / SAHI acquire first,
exactly as a fovea needs a saccade target.  The prior σ given to the network is the tracker's predictive std,
clipped to the training range.  Returns (cx, cy, w, h, score); the blur (length, axis angle) of the last call is kept
in .last_blur for the tracker's velocity measurement.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from .fovea import S, build, decode, make_input


class FoveaEye:
    def __init__(self, seq, ckpt: str, device: str = "cuda", k: int = 3, thr: float = 0.05):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.args = ck["args"]
        self.model, _ = build(self.args["arch"], self.args["frames"], not self.args["no_diff"], not self.args["no_prior"])
        self.model.load_state_dict(ck["model"]); self.model.to(device).eval()
        self.seq, self.dev, self.k, self.thr = seq, device, k, thr
        self.buf = {}; self.last_blur = []

    def _img(self, f):
        if f not in self.buf:
            self.buf[f] = cv2.cvtColor(cv2.imread(str(self.seq.img_path(max(1, f)))), cv2.COLOR_BGR2RGB)
            for q in [q for q in self.buf if q < f - 2]:
                del self.buf[q]
        return self.buf[f]

    @torch.no_grad()
    def __call__(self, f, xp=None, Pp=None, trk=None) -> List[tuple]:
        self._img(f)
        if xp is None or (trk is not None and trk.lost):
            self.last_blur = []; return []
        W, H = self.seq.width, self.seq.height
        x0 = int(np.clip(round(xp[0] - S / 2), 0, W - S)); y0 = int(np.clip(round(xp[1] - S / 2), 0, H - S))
        crops = [self._img(q)[y0:y0 + S, x0:x0 + S] for q in (f - 2, f - 1, f)]
        x = torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0).permute(0, 3, 1, 2)[None].to(self.dev)
        sig = float(np.clip(np.sqrt(np.max(np.linalg.eigvalsh(Pp[:2, :2]))), 6.0, 48.0))
        prior = torch.tensor([[xp[0] - x0, xp[1] - y0]], dtype=torch.float32, device=self.dev)
        sg = torch.tensor([sig], device=self.dev)
        inp, mm = make_input(x, prior, sg, self.args["frames"], not self.args["no_diff"], not self.args["no_prior"])
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(inp, prior, sg, mm) if self.args["arch"] == "kpa" else self.model(inp)
        dets = decode({k: v.float() for k, v in out.items()}, k=self.k, thr=self.thr)[0]
        self.last_blur = [(x0 + d[0], y0 + d[1], d[5], d[6]) for d in dets]
        return [(x0 + d[0], y0 + d[1], d[2], d[3], d[4], d[5], d[6]) for d in dets]   # + blur length, axis angle
