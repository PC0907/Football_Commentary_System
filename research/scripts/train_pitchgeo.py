"""Train PitchGeo on SoccerNet-Calibration-2023 (runs locally for smoke tests, on Kaggle for real).

  python scripts/train_pitchgeo.py --train-zip .../train.zip --valid-zip .../valid.zip \
      --epochs 20 --bs 8 --out outputs/pitchgeo/run1 [--limit 64 --steps 20]   # smoke test

Writes: <out>/last.pt, <out>/best.pt, <out>/log.jsonl (per-epoch metrics), <out>/config.json
Validation metric: mean point-to-marking distance in pixels at 960x540 for present classes,
and presence F1 — a fast proxy; the official SN metric is computed by eval_pitchgeo.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.calib.data import SNCalibZip  # noqa: E402
from fcs.calib.model import PitchGeo, pitchgeo_loss  # noqa: E402
from fcs.calib.pitch import ALL_CLASSES, CIRCLE_CLASSES  # noqa: E402
from fcs.progress import Progress  # noqa: E402

LINE_IDX = [i for i, n in enumerate(ALL_CLASSES) if n not in CIRCLE_CLASSES]
CIRC_IDX = [i for i, n in enumerate(ALL_CLASSES) if n in CIRCLE_CLASSES]


def run_epoch(model, loader, opt, scaler, device, train, prog, sched=None, max_steps=None):
    model.train(train)
    agg, n = {}, 0
    tp = fp = fn = 0
    for step, b in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"), torch.set_grad_enabled(train):
            out = model(b["image"])
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        loss, st = pitchgeo_loss(out, b, LINE_IDX, CIRC_IDX, probabilistic=not ARGS.deterministic)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if sched:
                sched.step()
        pred = (out["presence"] > 0).float()
        tp += float((pred * b["present"]).sum()); fp += float((pred * (1 - b["present"])).sum())
        fn += float(((1 - pred) * b["present"]).sum())
        for k, v in st.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        prog.update(1)
    res = {k: v / max(n, 1) for k, v in agg.items()}
    res["presence_f1"] = 2 * tp / max(1.0, 2 * tp + fp + fn)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-zip", required=True)
    ap.add_argument("--valid-zip", required=True)
    ap.add_argument("--backbone", default="facebook/dinov2-small")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--size", type=int, nargs=2, default=[644, 364])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--steps", type=int, help="max steps per epoch (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pitchgeo" / "run"))
    ap.add_argument("--deterministic", action="store_true", help="Huber loss instead of Laplace NLL (ablation)")
    a = ap.parse_args()
    global ARGS
    ARGS = a
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    tr = SNCalibZip(a.train_zip, tuple(a.size), augment=True, limit=a.limit)
    va = SNCalibZip(a.valid_zip, tuple(a.size), augment=False, limit=a.limit)
    dl_tr = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(va, a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    model = PitchGeo(a.backbone).to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": rest, "lr": a.lr}], weight_decay=0.05)
    steps_per_epoch = min(len(dl_tr), a.steps or 10 ** 9)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr], total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    va_steps = min(len(dl_va), a.steps or 10 ** 9)
    prog = Progress(a.epochs * (steps_per_epoch + va_steps), "train PitchGeo", "batch")
    best = float("inf")
    for ep in range(a.epochs):
        t0 = time.time()
        r_tr = run_epoch(model, dl_tr, opt, scaler, device, True, prog, sched, a.steps)
        r_va = run_epoch(model, dl_va, opt, scaler, device, False, prog, None, a.steps)
        score = r_va["line_px540"] + r_va["ell_px540"]
        rec = dict(epoch=ep, time_s=time.time() - t0, train=r_tr, valid=r_va)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"ep {ep}: train loss parts {r_tr['line_px540']:.1f}px/{r_tr['ell_px540']:.1f}px | valid line "
              f"{r_va['line_px540']:.2f}px ellipse {r_va['ell_px540']:.2f}px presF1 {r_va['presence_f1']:.3f} sigma {r_va['sigma_px540']:.1f}px", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "last.pt")
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "best.pt")
    prog.close()


if __name__ == "__main__":
    main()
