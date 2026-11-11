"""Fine-tune a person/ball detector on SoccerNet-Tracking train and write test detections.

  python scripts/train_detector_snmot.py --arch yolo26m --snmot ROOT --out OUT [--stride 5 --epochs 40]
  python scripts/train_detector_snmot.py --arch rfdetr-m --snmot ROOT --out OUT

Data: SNMOT train sequences; every `stride`-th frame; classes from gameinfo roles:
  0 player, 1 goalkeeper, 2 referee, 3 ball  ("other" tracklets ignored).
Validation = the 15-sequence tracker-tuning subset (every 4th train sequence) — never test.
After training: inference on every frame of every TEST sequence (conf >= 0.05) written as
  OUT/dets/test/<seq>.txt  rows: frame,-1,x,y,w,h,score,role,-1,-1   (cache format of
  scripts/precompute_dets.py, so the trackers can use it directly).
"""
import argparse
import configparser
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROLE = {"player": 0, "goalkeeper": 1, "goalkeepers": 1, "referee": 2, "ball": 3}
NAMES = ["player", "goalkeeper", "referee", "ball"]


def roles(seq_dir):
    g = configparser.ConfigParser(); g.read(seq_dir / "gameinfo.ini")
    out = {}
    for k, v in g["Sequence"].items():
        if k.startswith("trackletid_"):
            r = v.strip().lower().split()[0].split(";")[0]
            if r in ROLE:
                out[int(k.split("_")[1])] = ROLE[r]
    return out


def seq_info(seq_dir):
    s = configparser.ConfigParser(); s.read(seq_dir / "seqinfo.ini")
    s = s["Sequence"]
    return int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"])


def build_yolo(snmot, work, stride, val_seqs, max_seqs=None):
    """YOLO txt layout: images/{train,val}, labels/{train,val} (symlinked images)."""
    for split in ("train", "val"):
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in (snmot / "train").iterdir() if p.is_dir())
    if max_seqs:                                   # smoke test: a few train + val sequences
        seqs = [q for q in seqs if q.name not in val_seqs][:max_seqs] + [q for q in seqs if q.name in val_seqs][:1]
    n = {"train": 0, "val": 0}
    for sd in seqs:
        split = "val" if sd.name in val_seqs else "train"
        L, W, H = seq_info(sd)
        rmap = roles(sd)
        gt = np.loadtxt(sd / "gt" / "gt.txt", delimiter=",", ndmin=2)
        for f in range(1, L + 1, stride if split == "train" else 3 * stride):
            rows = gt[gt[:, 0] == f]
            name = f"{sd.name}_{f:06d}"
            os.symlink(sd / "img1" / f"{f:06d}.jpg", work / "images" / split / f"{name}.jpg")
            with open(work / "labels" / split / f"{name}.txt", "w") as fh:
                for r in rows:
                    c = rmap.get(int(r[1]))
                    if c is None:
                        continue
                    x, y, w, h = r[2:6]
                    x0, y0 = max(0.0, x), max(0.0, y); x1, y1 = min(W, x + w), min(H, y + h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    fh.write(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
            n[split] += 1
    (work / "data.yaml").write_text(
        f"path: {work}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {nm}\n" for i, nm in enumerate(NAMES)))
    print("dataset:", n, flush=True)
    return work / "data.yaml"


def build_coco(yolo_dir, coco_dir):
    """Roboflow-style COCO folders (train/valid/test with _annotations.coco.json) for RF-DETR."""
    from PIL import Image
    for split, dst in (("train", "train"), ("val", "valid"), ("val", "test")):
        out = coco_dir / dst; out.mkdir(parents=True, exist_ok=True)
        imgs, anns = [], []
        for k, ip in enumerate(sorted((yolo_dir / "images" / split).glob("*.jpg"))):
            os.symlink(os.path.realpath(ip), out / ip.name)
            W, H = Image.open(ip).size
            imgs.append(dict(id=k, file_name=ip.name, width=W, height=H))
            for line in open(yolo_dir / "labels" / split / f"{ip.stem}.txt"):
                c, cx, cy, w, h = map(float, line.split())
                anns.append(dict(id=len(anns), image_id=k, category_id=int(c), iscrowd=0,
                                 bbox=[(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H], area=w * W * h * H))
        cats = [dict(id=i, name=nm, supercategory="none") for i, nm in enumerate(NAMES)]
        (out / "_annotations.coco.json").write_text(json.dumps(dict(images=imgs, annotations=anns, categories=cats)))


def infer_test(predict_fn, snmot, out_dir, batch=8, max_seqs=None, max_frames=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for sd in sorted(p for p in (snmot / "test").iterdir() if p.is_dir())[:max_seqs]:
        L, _, _ = seq_info(sd)
        L = min(L, max_frames or L)
        rows = []
        paths = [str(sd / "img1" / f"{f:06d}.jpg") for f in range(1, L + 1)]
        for i in range(0, L, batch):
            for k, dets in enumerate(predict_fn(paths[i:i + batch])):
                for x0, y0, x1, y1, s, c in dets:
                    rows.append(f"{i + k + 1},-1,{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},{s:.4f},{int(c)},-1,-1")
        (out_dir / f"{sd.name}.txt").write_text("\n".join(rows) + "\n")
        print(sd.name, len(rows), flush=True)


def _rf_train(cls, imgsz, kw):
    """RF-DETR training in its own process group so a wall-clock budget can stop it cleanly."""
    os.setsid()
    import rfdetr
    getattr(rfdetr, cls)(resolution=imgsz).train(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="yolo26m")
    ap.add_argument("--snmot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/kaggle/temp/det")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="local end-to-end check: 2 train seqs, 1 epoch, 1 test seq x 16 frames")
    ap.add_argument("--smoke-epochs", type=int, default=1)
    ap.add_argument("--smoke-stride", type=int, default=50)
    ap.add_argument("--train-hours", type=float, default=7.0,
                    help="hard wall-clock cap on training so test inference always runs (Kaggle kills at 12 h)")
    a = ap.parse_args()
    ap_smoke_epochs = a.smoke_epochs
    if a.smoke:
        a.epochs, a.stride = ap_smoke_epochs, a.smoke_stride
    snmot, out, work = Path(a.snmot), Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in (snmot / "train").iterdir() if p.is_dir())
    val_seqs = set(seqs[::4])                                   # same tuning subset as the trackers
    if work.exists():
        shutil.rmtree(work)
    data = build_yolo(snmot, work / "yolo", a.stride, val_seqs, max_seqs=2 if a.smoke else None)
    if a.arch.startswith("yolo"):
        from ultralytics import YOLO
        model = YOLO(f"{a.arch}.pt")
        model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=10,
                    project=str(out), name="train", exist_ok=True, close_mosaic=5, workers=4,
                    time=a.train_hours)   # ultralytics: `time` (h) overrides epochs
        best = YOLO(str(out / "train" / "weights" / "best.pt"))
        metrics = best.val(data=str(data), imgsz=a.imgsz, batch=a.batch)
        (out / "val_metrics.json").write_text(json.dumps({k: float(v) for k, v in metrics.results_dict.items()}, indent=1))

        def predict(paths):
            res = best(paths, imgsz=a.imgsz, conf=0.05, verbose=False, half=True)
            return [[(*b, s, c) for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                    r.boxes.cls.cpu().numpy())] for r in res]
    elif a.arch.startswith("rfdetr"):
        import rfdetr
        build_coco(work / "yolo", work / "coco")
        cls = {"rfdetr-s": "RFDETRSmall", "rfdetr-m": "RFDETRMedium", "rfdetr-b": "RFDETRBase"}[a.arch]
        assert a.imgsz % 32 == 0, "RF-DETR resolution must be divisible by patch_size*num_windows=32"
        import multiprocessing as mp
        import signal
        import time as _t
        kw = dict(dataset_dir=str(work / "coco"), epochs=a.epochs, batch_size=max(1, a.batch // 2),
                  grad_accum_steps=4, output_dir=str(out / "train"), early_stopping=True)
        proc = mp.get_context("spawn").Process(target=_rf_train, args=(cls, a.imgsz, kw))
        t0 = _t.time(); proc.start(); proc.join(a.train_hours * 3600)
        if proc.is_alive():
            print(f"training budget {a.train_hours} h reached -> stopping, using best checkpoint so far", flush=True)
            os.killpg(proc.pid, signal.SIGTERM); proc.join(120)
            if proc.is_alive():
                os.killpg(proc.pid, signal.SIGKILL); proc.join()
        elif proc.exitcode != 0:
            sys.exit(f"RF-DETR training failed (exit {proc.exitcode})")
        print(f"training wall-clock {(_t.time() - t0) / 3600:.2f} h", flush=True)
        ck = sorted((out / "train").glob("checkpoint_best*.pth"))
        assert ck, "no RF-DETR checkpoint saved"
        model = getattr(rfdetr, cls)(pretrain_weights=str(ck[0]), resolution=a.imgsz)
        print("RF-DETR checkpoint:", ck[0], flush=True)

        def predict(paths):
            from PIL import Image
            ds = model.predict([Image.open(p) for p in paths], threshold=0.05)
            ds = ds if isinstance(ds, list) else [ds]
            # RF-DETR has a (num_classes+1)-th "no-object" slot that can surface at low thresholds
            return [[(*b, s, c) for b, s, c in zip(d.xyxy, d.confidence, d.class_id) if 0 <= c < len(NAMES)] for d in ds]
    else:
        sys.exit("unknown arch")
    infer_test(predict, snmot, out / "dets" / "test", max_seqs=1 if a.smoke else None, max_frames=16 if a.smoke else None)
    n = sum(1 for f in (out / "dets" / "test").glob("*.txt") for _ in open(f))
    print("test detections written:", n, flush=True)
    assert n > 0, "no detections written"


if __name__ == "__main__":
    main()
