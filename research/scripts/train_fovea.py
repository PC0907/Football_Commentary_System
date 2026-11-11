"""Train the fovea network on crops from scripts/extract_ball_crops.py (Kaggle output ball_crops.zip).
  python scripts/train_fovea.py --zip outputs/ball/crops/ball_crops.zip --arch unet --out outputs/ball/fovea/unet
Ablations: --frames 1 | --no-diff | --no-prior.  Validation: 4 held-out SEQUENCES (by name), never seen in training.
Metrics (val): hit@5 = top peak within 5 px of the GT (positives); best-F1 over the score threshold, where a
negative crop with a peak above the threshold is a false positive; per-speed hit@5."""
import argparse, io, json, math, sys, time, zipfile
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fcs.ball.fovea import S, STRIDE, build, decode, focal_loss, make_input  # noqa: E402
from fcs.progress import Progress  # noqa: E402


class Crops(Dataset):
    def __init__(self, zp, labels, train):
        self.zp, self.L, self.train, self.z = zp, labels, train, None

    def __len__(self):
        return len(self.L)

    def __getitem__(self, i):
        if self.z is None:
            self.z = zipfile.ZipFile(self.zp)
        l = self.L[i]; rng = np.random
        ims = [cv2.cvtColor(cv2.imdecode(np.frombuffer(self.z.read(f"img/{l['key']}_{k}.jpg"), np.uint8), 1), cv2.COLOR_BGR2RGB) for k in range(3)]
        full = ims[0].shape[0]
        px, py = l["prior"]
        # 160 window inside the 192 crop, centred on the prior (as at inference), with small jitter
        ox = int(np.clip(round(px - S / 2 + (rng.randint(-8, 9) if self.train else 0)), 0, full - S))
        oy = int(np.clip(round(py - S / 2 + (rng.randint(-8, 9) if self.train else 0)), 0, full - S))
        ims = [im[oy:oy + S, ox:ox + S] for im in ims]
        prior = np.array([px - ox, py - oy], np.float32)
        vis = l["vis"]; c = np.array(l["c"], np.float32) - [ox, oy] if vis else None
        if vis and not (0 <= c[0] < S and 0 <= c[1] < S):
            vis = 0
        v = np.array(l["v"], np.float32) if vis and l.get("v") else None
        if self.train and rng.rand() < 0.5:                                    # horizontal flip
            ims = [im[:, ::-1] for im in ims]; prior[0] = S - 1 - prior[0]
            if vis: c[0] = S - 1 - c[0]
            if v is not None: v[0] = -v[0]
        x = np.stack(ims).astype(np.float32) / 255.0                           # 3,S,S,3
        if self.train:
            x = np.clip(x * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08), 0, 1)
        g = S // STRIDE; hm = np.zeros((g, g), np.float32); reg = np.zeros(8, np.float32)  # off2 size2 blur2 mask_box mask_blur
        cell = np.array([0, 0])
        if vis:
            cg = c / STRIDE; cell = np.clip(cg.astype(int), 0, g - 1)
            yy, xx = np.mgrid[:g, :g]; hm = np.exp(-((xx - cg[0]) ** 2 + (yy - cg[1]) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
            hm[cell[1], cell[0]] = 1.0
            w, h = l["wh"]; reg[:4] = [cg[0] - cell[0], cg[1] - cell[1], math.log(max(w, 2)), math.log(max(h, 2))]; reg[6] = 1
            if v is not None:
                sp = float(np.hypot(*v)); th = math.atan2(v[1], v[0]); reg[4:6] = [sp * math.cos(2 * th), sp * math.sin(2 * th)]; reg[7] = 1
        sp = float(np.hypot(*v)) if v is not None else -1.0
        return dict(x=torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), prior=torch.from_numpy(prior), sigma=torch.tensor(float(l["sigma"])),
                    hm=torch.from_numpy(hm[None]), reg=torch.from_numpy(reg), cell=torch.from_numpy(cell), vis=torch.tensor(vis),
                    c=torch.from_numpy(c if vis else np.zeros(2, np.float32)), speed=torch.tensor(sp))


def losses(out, b):
    lh = focal_loss(out["hm"], b["hm"])
    B = b["hm"].shape[0]; idx = torch.arange(B, device=b["hm"].device); cx, cy = b["cell"][:, 0], b["cell"][:, 1]
    g = lambda k: out[k][idx, :, cy, cx]                                       # B×2 at the GT cell
    m_box, m_blur = b["reg"][:, 6:7], b["reg"][:, 7:8]
    l_off = (torch.abs(g("off") - b["reg"][:, 0:2]) * m_box).sum() / m_box.sum().clamp(min=1)
    l_size = (torch.abs(g("size") - b["reg"][:, 2:4]) * m_box).sum() / m_box.sum().clamp(min=1)
    l_blur = (torch.abs(g("blur") - b["reg"][:, 4:6]) * m_blur).sum() / m_blur.sum().clamp(min=1) / 10.0
    return lh + l_off + 0.5 * l_size + 0.2 * l_blur, dict(hm=float(lh), off=float(l_off), size=float(l_size), blur=float(l_blur))


@torch.no_grad()
def evaluate(model, dl, a, dev):
    model.eval(); recs = []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        x, mm = make_input(b["x"], b["prior"], b["sigma"], a.frames, not a.no_diff, not a.no_prior)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, b["prior"], b["sigma"], mm) if a.arch == "kpa" else model(x)
        out = {k: v.float() for k, v in out.items()}
        for i, d in enumerate(decode(out, k=1, thr=0.0)):
            s = d[0][4] if d else 0.0
            err = math.hypot(d[0][0] - float(b["c"][i, 0]), d[0][1] - float(b["c"][i, 1])) if d and b["vis"][i] else 1e9
            recs.append((int(b["vis"][i]), s, err, float(b["speed"][i])))
    R = np.array(recs); pos = R[R[:, 0] == 1]; neg = R[R[:, 0] == 0]
    hit = pos[:, 2] <= 5
    best = (0, 0)
    for t in np.linspace(0.05, 0.95, 19):
        tp = np.sum(hit & (pos[:, 1] >= t)); fp = np.sum((pos[:, 1] >= t) & ~hit) + np.sum(neg[:, 1] >= t); fn = len(pos) - tp
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); best = max(best, (f1, t))
    bins = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 1e9)]
    by = {f"{lo}-{hi if hi < 1e8 else 'inf'}": round(float(np.mean(hit[(pos[:, 3] >= lo) & (pos[:, 3] < hi)])), 3)
          for lo, hi in bins if np.any((pos[:, 3] >= lo) & (pos[:, 3] < hi))}
    model.train()
    return dict(hit5=round(float(hit.mean()), 4), bestF1=round(float(best[0]), 4), thr=round(float(best[1]), 2),
                neg_max_p50=round(float(np.median(neg[:, 1])) if len(neg) else 0.0, 3), hit5_by_speed=by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--out", required=True); ap.add_argument("--arch", default="unet")
    ap.add_argument("--frames", type=int, default=3); ap.add_argument("--no-diff", action="store_true"); ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--n-val-seqs", type=int, default=4); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip); L = [json.loads(l) for l in z.read("labels.jsonl").decode().splitlines()]
    seqs = sorted({l["seq"] for l in L}); val_seqs = set(seqs[:: max(1, len(seqs) // a.n_val_seqs)][:a.n_val_seqs])
    tr = [l for l in L if l["seq"] not in val_seqs]; va = [l for l in L if l["seq"] in val_seqs]
    if a.limit: tr, va = tr[:a.limit], va[: max(200, a.limit // 5)]
    print(f"train {len(tr)} val {len(va)} (val seqs {sorted(val_seqs)})", flush=True)
    dev = "cuda"
    model, c_in = build(a.arch, a.frames, not a.no_diff, not a.no_prior); model = model.to(dev)
    dl = DataLoader(Crops(a.zip, tr, True), a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    dv = DataLoader(Crops(a.zip, va, False), 64, shuffle=False, num_workers=a.workers)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=max(0.1, 3.0 / (a.epochs * len(dl))))
    scaler = torch.amp.GradScaler(); prog = Progress(a.epochs * len(dl), f"fovea {a.arch} f{a.frames}{'' if not a.no_diff else ' -diff'}{'' if not a.no_prior else ' -prior'}", "step")
    best = -1; log = open(out / "log.jsonl", "w")
    for ep in range(a.epochs):
        t0 = time.time(); agg = {}
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            x, mm = make_input(b["x"], b["prior"], b["sigma"], a.frames, not a.no_diff, not a.no_prior)
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(x, b["prior"], b["sigma"], mm) if a.arch == "kpa" else model(x)
            loss, parts = losses({k: v.float() for k, v in o.items()}, b)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            for k, v in parts.items(): agg[k] = agg.get(k, 0) + v / len(dl)
            prog.update(1)
        ev = evaluate(model, dv, a, dev)
        rec = dict(epoch=ep, train=agg, val=ev, minutes=round((time.time() - t0) / 60, 2)); log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if ev["bestF1"] > best:
            best = ev["bestF1"]
            torch.save(dict(model=model.state_dict(), args=vars(a), val=ev, c_in=c_in), out / "best.pt")
    prog.close()


if __name__ == "__main__":
    main()
