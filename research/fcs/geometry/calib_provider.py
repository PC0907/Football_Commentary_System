"""Per-sequence calibration under the calibration variants (none / c0 / c1 / c1L / fused / smooth ...).

  c0         production RANSAC fit per frame
  c1         robust weighted fit + uncertainty, per frame
  fused      C2: forward keypoint ⊕ ego-motion filter (online)
  smooth     C2-smooth: forward/backward two-filter smoother (offline)
Suffix "L" (c1L, fusedL, smoothL) or mode "c1l": use the point-on-line sensor
(calibrate_c1l) instead of the field-point-intersection sensor.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .camera import CalibrationFilter, smooth_calibrations
from .keypoint_calib import calibrate_c0, calibrate_c1, calibrate_c1l

DEFAULTS = dict(drift_m=0.15, gate=4.0, kp_floor=0.5)


def sequence_calibrations(mode: str, kp: np.ndarray, M: np.ndarray, width: int, height: int,
                          **kw) -> List[Tuple[Optional[np.ndarray], float]]:
    T = len(kp) - 1
    p = {**DEFAULTS, **{k: v for k, v in kw.items() if k != "sensor"}}
    if mode == "none":
        return [(None, np.inf)] * (T + 1)
    if mode == "c0":
        return [(None, np.inf)] + [calibrate_c0(kp[t]) for t in range(1, T + 1)]
    sensor = kw.pop("sensor", None) or ("lines" if mode.endswith("L") or mode in ("c1l",) else "points")
    mode = mode.rstrip("L") if mode != "c1l" else "c1"
    fn = calibrate_c1l if sensor == "lines" else calibrate_c1
    meas = [(None, np.inf)] + [fn(kp[t]) for t in range(1, T + 1)]
    if mode == "c1":
        return meas
    Hk = [m[0] for m in meas]
    sk = [m[1] for m in meas]
    if mode == "fused":
        cf = CalibrationFilter(width, height, **p)
        out = [(None, np.inf)]
        for t in range(1, T + 1):
            c = cf.step(M[t], Hk[t], sk[t])
            out.append((c.H, c.sigma_m))
        return out
    if mode == "smooth":
        cs = smooth_calibrations(Hk, sk, M, width, height, **p)
        return [(None, np.inf)] + [(c.H, c.sigma_m) for c in cs[1:]]
    raise ValueError(mode)
