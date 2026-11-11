"""Push a Kaggle kernel when a GPU slot is free (retrying), optionally only after another
named push has happened (to respect the orchestrator's priorities); then watch it.
  python scripts/kaggle_queue_push.py TASK NAME "ARGS" OUT_DIR --after push-fcs-pitchgeo-deterministic --expect-min 240"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.progress import Progress
ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("name"); ap.add_argument("args"); ap.add_argument("out")
ap.add_argument("--after"); ap.add_argument("--expect-min", type=int, default=240)
a = ap.parse_args()
prog = Progress(1, f"queue {a.name}: waiting for {a.after or 'a free slot'}", "push")
st = ROOT / "logs" / "overnight_status.json"
while a.after and not (st.exists() and a.after in json.loads(st.read_text())):
    time.sleep(120)
while True:
    r = subprocess.run([sys.executable, str(ROOT / "kaggle/push_kernel.py"), a.task, a.name, a.args],
                       capture_output=True, text=True, cwd=ROOT)
    if "successfully pushed" in r.stdout:
        break
    prog.update(0, desc=f"queue {a.name}: slot busy, retry in 10 min")
    time.sleep(600)
prog.update(1); prog.close()
time.sleep(60)
sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts/kaggle_watch.py"), f"gk8247/{a.name}", a.out,
                          "--expect-min", str(a.expect_min)]))
