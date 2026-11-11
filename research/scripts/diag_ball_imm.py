"""Why does the ball IMM (B2/B3) output so few frames?  Per sequence:
  frac_H      frames with a homography
  frac_cam    frames where camera_from_homography succeeds (the IMM needs this)
  frac_det    frames with a ball candidate >= init_score
  frac_out_*  frames each tracker outputs a box
Writes outputs/diag/ball_imm_<split>.json; prints the means.
  python scripts/diag_ball_imm.py --split train --det yolov8m_repo --calib fusedL --n 6
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.geometry.calib_provider import sequence_calibrations
from fcs.geometry.projection import camera_from_homography
from fcs.tracking.ball import BallConfig, make_ball_tracker
from fcs.tracking.data import SNMOT_ROOT, detections_by_frame, load_sequence
from fcs.progress import Progress
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--det", default="yolov8m_repo")
ap.add_argument("--calib", default="fusedL"); ap.add_argument("--n", type=int, default=6); ap.add_argument("--variants", nargs="+", default=["B0", "B2"])
a = ap.parse_args()
seqs = sorted(p.name for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())[:a.n]
rows = []; prog = Progress(len(seqs), "ball IMM diagnosis", "seq")
for s in seqs:
    seq = load_sequence(SNMOT_ROOT / a.split / s)
    c = np.load(ROOT / "cache/snmot" / a.split / f"{s}.npz")
    cal = sequence_calibrations(a.calib, c["kp"], c["M"], seq.width, seq.height)
    dets = detections_by_frame(np.loadtxt(ROOT / "cache/dets" / a.det / a.split / f"{s}.txt", delimiter=",", ndmin=2), seq.length)
    L = seq.length; r = dict(seq=s, H=0, cam=0, det=0)
    trk = {v: make_ball_tracker(BallConfig(variant=v, fps=seq.fps, img_w=seq.width, img_h=seq.height)) for v in a.variants}
    out = {v: 0 for v in a.variants}
    for f in range(1, L + 1):
        H = cal[f][0]; d = dets[f]
        balls = d[d[:, 5] == 3] if len(d) else d
        players = d[(d[:, 5] <= 1) & (d[:, 4] >= 0.4)] if len(d) else d
        r["H"] += H is not None
        r["cam"] += H is not None and camera_from_homography(H, seq.width, seq.height) is not None
        r["det"] += bool(len(balls)) and balls[:, 4].max() >= 0.35
        for v, t in trk.items():
            out[v] += t.step(f, balls, M=c["M"][f], H=H, players=players) is not None
    rows.append({"seq": s, "frac_H": r["H"] / L, "frac_cam": r["cam"] / L, "frac_det": r["det"] / L,
                 **{f"frac_out_{v}": out[v] / L for v in a.variants}}); prog.update(1)
prog.close()
mean = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "seq"}
Path(ROOT / "outputs/diag").mkdir(parents=True, exist_ok=True)
(ROOT / f"outputs/diag/ball_imm_{a.split}.json").write_text(json.dumps({"mean": mean, "per_seq": rows}, indent=1))
print(json.dumps({k: round(v, 3) for k, v in mean.items()}))
