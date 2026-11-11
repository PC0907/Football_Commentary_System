"""Run a fine-tuned YOLO detector locally on an SNMOT split -> cache/dets/<tag>/<split>/<seq>.txt
(same format as scripts/train_detector_snmot.py). The local 'train' split is the detector's held-out
validation subset (every 4th train sequence), so these detections are a legitimate tuning set.
  python scripts/infer_dets_local.py --weights outputs/detector/yolo26m_snmot/det/train/weights/best.pt \
      --split train --tag yolo26m_snmot
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.progress import Progress
from fcs.tracking.data import SNMOT_ROOT, load_sequence
ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--split", default="train")
ap.add_argument("--tag", required=True); ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()
from ultralytics import YOLO
model = YOLO(a.weights)
out = ROOT / "cache" / "dets" / a.tag / a.split; out.mkdir(parents=True, exist_ok=True)
seqs = sorted(p for p in (SNMOT_ROOT / a.split).iterdir() if p.is_dir())
prog = Progress(sum(load_sequence(s).length for s in seqs), f"YOLO inference {a.split}", "frame")
for sd in seqs:
    seq = load_sequence(sd); rows = []
    for i in range(1, seq.length + 1, a.batch):
        fr = list(range(i, min(i + a.batch, seq.length + 1)))
        for f, r in zip(fr, model([str(seq.img_path(f)) for f in fr], imgsz=a.imgsz, conf=0.05, verbose=False, half=True)):
            for (x0, y0, x1, y1), s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                rows.append(f"{f},-1,{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{s:.4f},{int(c)},-1,-1")
        prog.update(len(fr))
    (out / f"{sd.name}.txt").write_text("\n".join(rows) + "\n")
prog.close()
