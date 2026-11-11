"""Cache per-frame field keypoints and ego-motion homographies for SNMOT sequences.

Output: cache/snmot/<split>/<seq>.npz with
  kp  : (T+1, 46, 3)   field keypoints (x, y, conf) in full-resolution pixels (index = frame)
  M   : (T+1, 3, 3)    homography mapping frame t-1 pixels -> frame t pixels
  nM  : (T+1,)         RANSAC inliers of M
Usage: python scripts/precompute_snmot.py --split test [--seqs SNMOT-116 ...]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.geometry.camera import EgoMotion  # noqa: E402
from fcs.geometry.keypoint_calib import keypoints_model  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence  # noqa: E402
from fcs.progress import Progress  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    out_dir = ROOT / "cache" / "snmot" / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    model = keypoints_model()
    seq_dirs = sorted(p for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
    if a.seqs:
        seq_dirs = [p for p in seq_dirs if p.name in a.seqs]
    pending = [sd for sd in seq_dirs if not (out_dir / f"{sd.name}.npz").exists()]
    prog = Progress(sum(load_sequence(sd).length for sd in pending), f"keypoints+ego-motion {a.split}", "frame")
    for sd in pending:
        out = out_dir / f"{sd.name}.npz"
        seq = load_sequence(sd)
        dets = detections_by_frame(seq.det(), seq.length)
        T = seq.length
        kp = np.zeros((T + 1, 46, 3), np.float32)
        M = np.tile(np.eye(3), (T + 1, 1, 1))
        nM = np.zeros(T + 1, np.int32)
        ego = EgoMotion()
        t0 = time.time()
        batch, idx = [], []
        for f in range(1, T + 1):
            img = cv2.imread(str(seq.img_path(f)))
            M[f], nM[f] = ego(img, dets[f][:, :4] if len(dets[f]) else None)
            batch.append(img); idx.append(f)
            prog.update(1, desc=f"keypoints+ego-motion {a.split} {sd.name}")
            if len(batch) == a.batch or f == T:
                res = model(batch, verbose=False, half=True)
                for r, fi in zip(res, idx):
                    if r.keypoints is not None and len(r.keypoints.data):
                        k = r.keypoints.data.cpu().numpy()
                        # one instance ("footballField"); take the most confident one
                        best = int(np.argmax(r.boxes.conf.cpu().numpy())) if r.boxes is not None and len(r.boxes) else 0
                        kp[fi] = k[best]
                batch, idx = [], []
        np.savez_compressed(out, kp=kp, M=M, nM=nM)
        print(f"{sd.name}: {T} frames in {time.time() - t0:.0f}s", flush=True)
    prog.close()


if __name__ == "__main__":
    main()
