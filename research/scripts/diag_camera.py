"""Diagnostic: how often can a full camera be recovered from the calibration, and how
consistent is it with the homography (ground-point reprojection error in pixels)?"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcs.geometry.calib_provider import sequence_calibrations
from fcs.geometry.projection import camera_from_homography
from fcs.geometry.camera import apply_h
from fcs.tracking.data import SNMOT_ROOT, load_sequence
from fcs.progress import Progress
split, mode = sys.argv[1], sys.argv[2]
names = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "cache/snmot" / split).glob("*.npz"))
n_h = n_cam = 0; errs = []; heights = []
P = np.array([[x, y] for x in (20, 52.5, 85) for y in (10, 34, 58)], float)
prog = Progress(len(names), f"camera recovery {split}/{mode}", "seq")
for name in names:
    seq = load_sequence(SNMOT_ROOT / split / name); c = np.load(f"cache/snmot/{split}/{name}.npz")
    cal = sequence_calibrations(mode, c["kp"], c["M"], seq.width, seq.height)
    for t in range(1, seq.length + 1, 5):
        H = cal[t][0]
        if H is None: continue
        n_h += 1
        cam = camera_from_homography(H, seq.width, seq.height, max_resid_px=1e9)
        if cam is None: continue
        n_cam += 1
        u1 = cam.project(np.c_[P, np.zeros(len(P))]); u2 = apply_h(np.linalg.inv(H), P)
        errs.append(getattr(cam, 'resid_px', np.nan)); heights.append(-cam.center[2])
    prog.update(1)
prog.close()
errs = np.array(errs)
print(f"{split}/{mode}: frames with H {n_h}, camera recovered {n_cam} ({100*n_cam/max(1,n_h):.1f}%), "
      f"fit RMS residual (visible grid) px p10 {np.percentile(errs,10):.1f} median {np.median(errs):.1f} p90 {np.percentile(errs,90):.1f}, "
      f"frac<5px {np.mean(errs<5):.2f} <10px {np.mean(errs<10):.2f} <20px {np.mean(errs<20):.2f}, camera height median {np.median(heights):.1f} m")
