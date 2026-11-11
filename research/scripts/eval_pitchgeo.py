"""Evaluate a trained PitchGeo checkpoint on SoccerNet-Calibration-2023 (valid / test).

Stage 1 (GPU): network inference on every image -> raw predictions cached as .npz.
Stage 2 (CPU, parallel): MAP camera solve per image (+ k1 distortion, priors, mirror
hypotheses) and the re-implemented SN metric (completeness, JaC@5/10/20, final score).

  python scripts/eval_pitchgeo.py --ckpt outputs/pitchgeo/kaggle_run1/run/best.pt \
      --zip ~/soccerNet/hf/calib2023/valid.zip --tag K3_valid [--p-thr 0.5] [--no-priors]
"""
import argparse
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.calib.data import SNCalibZip  # noqa: E402
from fcs.calib.model import PitchGeo  # noqa: E402
from fcs.calib.sn_metric import H_EVAL, W_EVAL, evaluate_image, summarise  # noqa: E402
from fcs.calib.solve import from_network, solve_camera  # noqa: E402
from fcs.progress import Progress  # noqa: E402

KEYS = ("presence", "log_sigma", "n", "rho", "centre", "axes", "orient")


def infer(ckpt, zip_path, cache, bs=8, limit=None):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    model = PitchGeo(args.get("backbone", "facebook/dinov2-small")).cuda().eval()
    model.load_state_dict(ck["model"])
    ds = SNCalibZip(str(zip_path), tuple(args.get("size", (644, 364))), augment=False, limit=limit)
    dl = DataLoader(ds, bs, shuffle=False, num_workers=4)
    out = {k: [] for k in KEYS}; names = []
    prog = Progress(len(dl), f"PitchGeo inference {Path(zip_path).stem}", "batch")
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(b["image"].cuda())
            for k in KEYS:
                out[k].append(o[k].float().cpu().numpy())
            names += list(b["name"]); prog.update(1)
    prog.close()
    np.savez_compressed(cache, names=np.array(names), **{k: np.concatenate(v) for k, v in out.items()})


def work(args):
    i, zip_path, cache, p_thr, priors = args
    d = np.load(cache)
    o = {k: torch.from_numpy(d[k][i:i + 1]) for k in KEYS}
    obs = from_network(o, 0, W_EVAL, H_EVAL, p_thr=p_thr)
    sol = solve_camera(obs, priors=priors, distortion=True)
    ann = json.loads(zipfile.ZipFile(zip_path).read(str(d["names"][i])))
    return {t: evaluate_image(sol.camera, ann, t) for t in (5, 10, 20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--p-thr", type=float, default=0.5)
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out_dir = ROOT / "outputs" / "pitchgeo" / "eval"; out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{a.tag}_raw.npz"
    if not cache.exists():
        infer(a.ckpt, a.zip, cache, limit=a.limit)
    n = len(np.load(cache)["names"])
    prog = Progress(n, f"camera solve + metric {a.tag}", "img")
    accs = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(work, [(i, a.zip, str(cache), a.p_thr, not a.no_priors) for i in range(n)], chunksize=4):
            accs.append(r); prog.update(1)
    prog.close()
    res = summarise(accs)
    (out_dir / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(a.tag, json.dumps(res))


if __name__ == "__main__":
    main()
