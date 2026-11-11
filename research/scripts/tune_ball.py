"""Grid-search a ball tracker's parameters on the TUNING split (15 seqs, never test).
  python scripts/tune_ball.py --det yolo26m_snmot --variant B4 --grid s_ref=0.1,0.2 beta_abs=0.8,1.2 L=0,3
  python scripts/tune_ball.py --det yolo26m_snmot --variant B0 --grid init_score=0.2,0.3,0.35,0.45
Writes outputs/ball/train/<det>/tune_<variant><tag>.json (sorted by HOTA), prints the top 8."""
import argparse
import itertools
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from fcs.progress import Progress  # noqa: E402
from fcs.tracking.data import SNMOT_ROOT  # noqa: E402
from fcs.tracking.evaluate import evaluate  # noqa: E402
from run_ball import loc_metrics, run_one  # noqa: E402


def _run(args):
    s, out, c, variant, det, calib = args
    return run_one(s, split="train", variant=variant, det=det, calib=calib, out_root=out, overrides=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--variant", required=True)
    ap.add_argument("--grid", nargs="+", required=True); ap.add_argument("--calib", default="fusedL")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pv = lambda v: float(v) if "." in v else int(v)
    keys, vals = zip(*[(g.split("=")[0], [pv(x) for x in g.split("=")[1].split(",")]) for g in a.grid])
    configs = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    seqs = sorted(p.name for p in (SNMOT_ROOT / "train").iterdir() if p.is_dir())
    base = ROOT / "outputs" / "ball" / "train" / a.det / "tune" / (a.variant + a.tag)
    jobs = []
    for i, c in enumerate(configs):
        out = base / f"c{i:03d}"; out.mkdir(parents=True, exist_ok=True)
        jobs += [(s, out, c, a.variant, a.det, a.calib) for s in seqs]
    prog = Progress(len(jobs), f"tune ball {a.variant} ({len(configs)} configs)", "seq")
    with Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_run, jobs):
            prog.update(1)
    prog.close()
    dirs = {f"c{i:03d}": base / f"c{i:03d}" for i in range(len(configs))}
    res = evaluate(dirs, split="train", seqs=seqs, roles=[3], work=ROOT / "cache" / f"evalball_tune_{a.variant}")
    rows = []
    for k, r in res.items():
        r = {m: v for m, v in r.items() if not m.startswith("_")}
        r.update(loc_metrics("train", seqs, dirs[k]))
        rows.append(dict(config=configs[int(k[1:])], **r))
    rows.sort(key=lambda r: -r["HOTA"])
    (ROOT / "outputs" / "ball" / "train" / a.det / f"tune_{a.variant}{a.tag}.json").write_text(json.dumps(rows, indent=1))
    for r in rows[:8]:
        print(f"HOTA {r['HOTA']:.2f} R@10 {r['recall@10']:.1f} P@10 {r['prec@10']:.1f}  {r['config']}")
    shutil.rmtree(base)


if __name__ == "__main__":
    main()
