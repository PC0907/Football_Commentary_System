"""Detector fine-tune on Kaggle -> download test detections -> test-set tracking + ball tracking.
Every step is its own logged job (scripts/job.py).

  python scripts/job.py run chain-yolo26m -- python scripts/det_chain.py fcs-det-yolo26m yolo26m_snmot \
      "--arch yolo26m --epochs 40 --imgsz 1280 --batch 8 --train-hours 6" --expect-min 420
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
JOB = [PY, str(ROOT / "scripts" / "job.py"), "run"]
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("tag"); ap.add_argument("args")
ap.add_argument("--expect-min", type=int, default=420)
a = ap.parse_args()
out = ROOT / "outputs" / "detector" / a.tag


def step(name, cmd):
    rc = subprocess.call(JOB + [name, "--wait", "--"] + cmd, cwd=ROOT)
    print(f"[chain] {name}: rc {rc}", flush=True)
    return rc


# push when a GPU slot is free (retries every 10 min), then watch + download outputs
rc = step(f"kaggle-{a.tag}", [PY, "scripts/kaggle_queue_push.py", "detector", a.name, a.args, str(out),
                              "--expect-min", str(a.expect_min)])
dets = sorted(out.rglob("dets/test/*.txt"))
print(f"[chain] {len(dets)} detection files", flush=True)
if rc != 0 or len(dets) < 49:
    sys.exit(f"[chain] stopping: kaggle rc {rc}, {len(dets)}/49 detection files")
cache = ROOT / "cache" / "dets" / a.tag / "test"
cache.mkdir(parents=True, exist_ok=True)
for d in dets:
    (cache / d.name).write_text(d.read_text())
r1 = step(f"track-test-{a.tag}", [PY, "scripts/run_tracking.py", "--split", "test", "--det", a.tag, "--workers", "4",
                                  "--variants", "T0-fixed", "T1", "T2", "T3", "T4"])
r2 = step(f"ball-test-{a.tag}", [PY, "scripts/run_ball.py", "--split", "test", "--det", a.tag,
                                 "--variants", "B0", "B1", "--calib", "fusedL", "--workers", "4"])
sys.exit(r1 or r2)
