"""Cache detections of a YOLO detector on SNMOT frames (real-detector protocol).

Output: cache/dets/<name>/<split>/<seq>.txt in MOT rows:
  frame, -1, x, y, w, h, score, role, -1, -1     role: 0 player 1 GK 2 referee 3 ball 4 other
Low confidence threshold (0.05) so trackers can use low-score detections.

  python scripts/precompute_dets.py --split test --weights ../models/player_ball_detector_yolov8.pt --name yolov8m_repo
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.tracking.data import SNMOT_ROOT, load_sequence  # noqa: E402
from fcs.progress import Progress  # noqa: E402

# repo detector raw classes -> roles
REPO_MAP = {0: 0, 1: 1, 2: 3, 3: 2, 4: 2, 5: 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=str(ROOT.parent / "models" / "player_ball_detector_yolov8.pt"))
    ap.add_argument("--name", default="yolov8m_repo")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    model = YOLO(a.weights)
    cmap = REPO_MAP if a.name.endswith("_repo") else {i: i for i in range(5)}
    out_dir = ROOT / "cache" / "dets" / a.name / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [sd for sd in sorted(p for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
               if not (out_dir / f"{sd.name}.txt").exists()]
    prog = Progress(sum(load_sequence(sd).length for sd in pending), f"detector {a.name} {a.split}", "frame")
    for sd in pending:
        out = out_dir / f"{sd.name}.txt"
        seq = load_sequence(sd)
        rows = []
        paths = [str(seq.img_path(f)) for f in range(1, seq.length + 1)]
        for i in range(0, len(paths), a.batch):
            res = model(paths[i:i + a.batch], imgsz=a.imgsz, conf=a.conf, verbose=False, half=True)
            prog.update(len(res), desc=f"detector {a.name} {a.split} {sd.name}")
            for k, r in enumerate(res):
                f = i + k + 1
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), s, c in zip(xyxy, cf, cl):
                    rows.append((f, -1, x1, y1, x2 - x1, y2 - y1, s, cmap.get(int(c), 4)))
        with open(out, "w") as fh:
            for r in rows:
                fh.write(f"{r[0]},-1,{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.4f},{r[7]},-1,-1\n")
        print(sd.name, len(rows), flush=True)
    prog.close()


if __name__ == "__main__":
    main()
