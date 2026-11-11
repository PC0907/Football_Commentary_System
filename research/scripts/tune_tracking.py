"""Grid-search score-aware settings of our tracker on the TUNING split with real detections.
The tuning split (local 'train', 15 sequences) is the detector's held-out validation subset,
so neither the detector nor the tracker has seen these sequences' labels for training.

  python scripts/tune_tracking.py --det yolo26m_snmot --variant T1 \
      --grid det_high=0.3,0.5,0.6 min_score_new=0.5,0.6,0.7 det_low=0.1 min_track_len=1,10

Writes outputs/tracking/train/<det>/tune_<variant>.json (all configs, sorted by HOTA) and prints the top 10.
Output folders: outputs/tracking/train/<det>/tune/<variant>/<config-id>/ (removed afterwards unless --keep).
"""
import argparse
import itertools
import json
import shutil
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from fcs.progress import Progress  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT  # noqa: E402
from fcs.tracking.evaluate import evaluate  # noqa: E402
from run_tracking import run_one  # noqa: E402


def parse_val(v):
    return float(v) if "." in v else int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--variant", default="T1")
    ap.add_argument("--grid", nargs="+", required=True); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    keys, vals = zip(*[(g.split("=")[0], [parse_val(x) for x in g.split("=")[1].split(",")]) for g in a.grid])
    configs = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    seqs = sorted(p.name for p in (SNMOT_ROOT / "train").iterdir() if p.is_dir())
    base = ROOT / "outputs" / "tracking" / "train" / a.det / "tune" / a.variant
    jobs = []
    for i, c in enumerate(configs):
        out = base / f"c{i:03d}"; out.mkdir(parents=True, exist_ok=True)
        jobs += [(s, out, c) for s in seqs]
    prog = Progress(len(jobs), f"tune {a.variant} on {a.det} ({len(configs)} configs)", "seq")
    app_lr = ROOT / "cache" / "app_lr.json"

    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_run, [(s, o, c, a.variant, a.det, app_lr) for s, o, c in jobs]):
            prog.update(1)
    prog.close()
    res = evaluate({f"c{i:03d}": base / f"c{i:03d}" for i in range(len(configs))}, split="train", seqs=seqs,
                   roles=[0, 1, 2, 4], work=ROOT / "cache" / f"eval_tune_{a.det}_{a.variant}")
    rows = sorted([dict(config=configs[int(k[1:])], **{m: v for m, v in r.items() if not m.startswith("_")})
                   for k, r in res.items()], key=lambda r: -r["HOTA"])
    (ROOT / "outputs" / "tracking" / "train" / a.det / f"tune_{a.variant}.json").write_text(json.dumps(rows, indent=1))
    for r in rows[:10]:
        print(f"HOTA {r['HOTA']:.2f} AssA {r['AssA']:.2f} DetA {r['DetA']:.2f} IDF1 {r['IDF1']:.2f} IDSW {r['IDSW']}  {r['config']}")
    if not a.keep:
        shutil.rmtree(base)


def _run(args):
    s, out, c, variant, det, app_lr = args
    return run_one(s, split="train", variant=variant, det_mode=det, out_root=out, app_lr_path=app_lr, overrides=c)


if __name__ == "__main__":
    main()
