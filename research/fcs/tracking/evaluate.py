"""HOTA / CLEAR / Identity evaluation with the official TrackEval code.

Mirrors the SoccerNet-Tracking protocol (sn-tracking): MOTChallenge 2D box format,
class-agnostic, ``DO_PREPROC=False``. Optionally restricts GT and predictions to a
subset of GT roles (e.g. ball only) by writing filtered copies.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

_TE = Path(__file__).resolve().parents[2] / "third_party" / "TrackEval"
sys.path.insert(0, str(_TE))
# TrackEval predates NumPy 2: restore removed aliases instead of patching vendored code
for _a, _t in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
import trackeval  # noqa: E402

from .data import SNMOT_ROOT, load_sequence  # noqa: E402


def _prepare_gt(split: str, seqs: List[str], work: Path, roles: Optional[Iterable[int]]) -> Path:
    """Copy (optionally role-filtered) GT into a TrackEval-shaped folder."""
    gt_root = work / "gt" / split
    if gt_root.exists():
        shutil.rmtree(gt_root)
    for s in seqs:
        src = SNMOT_ROOT / split / s
        dst = gt_root / s
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "seqinfo.ini", dst / "seqinfo.ini")
        gt = np.loadtxt(src / "gt" / "gt.txt", delimiter=",", ndmin=2)
        if roles is not None:
            seq = load_sequence(src)
            keep = np.array([seq.role_of(i) in set(roles) for i in gt[:, 1]], bool)
            gt = gt[keep]
        np.savetxt(dst / "gt" / "gt.txt", gt, delimiter=",", fmt="%d")
    return gt_root


def evaluate(tracker_dirs: Dict[str, Path], split: str = "test", seqs: Optional[List[str]] = None,
             roles: Optional[Iterable[int]] = None, work: Path = Path("/tmp/fcs_eval"),
             per_seq: bool = False) -> Dict[str, dict]:
    """tracker_dirs: name -> folder containing <seq>.txt files.

    Returns {tracker: {metric: value}} for the COMBINED sequence summary.
    When ``roles`` is given, predictions are assumed to already be restricted to the
    corresponding objects (the caller filters by its own class labels).
    """
    if seqs is None:
        seqs = sorted(p.name for p in (SNMOT_ROOT / split).iterdir() if p.is_dir())
    work = Path(work)
    gt_root = _prepare_gt(split, seqs, work, roles)
    trk_root = work / "trackers" / split
    if trk_root.exists():
        shutil.rmtree(trk_root)
    for name, d in tracker_dirs.items():
        (trk_root / name / "data").mkdir(parents=True, exist_ok=True)
        for s in seqs:
            src = Path(d) / f"{s}.txt"
            dst = trk_root / name / "data" / f"{s}.txt"
            if src.exists():
                shutil.copy(src, dst)
            else:
                dst.write_text("")
    seqmap = work / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")

    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                   "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                   "USE_PARALLEL": True, "NUM_PARALLEL_CORES": 8})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trk_root),
                   "TRACKERS_TO_EVAL": list(tracker_dirs), "SEQMAP_FILE": str(seqmap),
                   "SKIP_SPLIT_FOL": True, "DO_PREPROC": False, "BENCHMARK": "SNMOT",
                   "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt", "PRINT_CONFIG": False})
    evaluator = trackeval.Evaluator(ev_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    out = {}
    for name in tracker_dirs:
        r = res["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
        out[name] = {
            "HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])),
            "DetA": 100 * float(np.mean(r["HOTA"]["DetA"])),
            "AssA": 100 * float(np.mean(r["HOTA"]["AssA"])),
            "LocA": 100 * float(np.mean(r["HOTA"]["LocA"])),
            "MOTA": 100 * float(r["CLEAR"]["MOTA"]),
            "IDF1": 100 * float(r["Identity"]["IDF1"]),
            "IDSW": int(r["CLEAR"]["IDSW"]),
            "Frag": int(r["CLEAR"]["Frag"]),
        }
        if per_seq:                       # for paired significance tests (scripts/bootstrap_tracking.py)
            out[name]["_per_seq"] = {
                s: {"HOTA": 100 * float(np.mean(q["HOTA"]["HOTA"])), "AssA": 100 * float(np.mean(q["HOTA"]["AssA"])),
                    "IDF1": 100 * float(q["Identity"]["IDF1"]), "IDSW": int(q["CLEAR"]["IDSW"])}
                for s in seqs for q in [res["MotChallenge2DBox"][name][s]["pedestrian"]]}
    return out
