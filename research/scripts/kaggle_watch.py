"""Watch a Kaggle kernel until it finishes, then download its outputs.
  python scripts/kaggle_watch.py KERNEL_SLUG OUT_DIR [--expect-min 150]
Progress = elapsed / expected duration (Kaggle exposes no live progress)."""
import argparse, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcs.progress import Progress
ap = argparse.ArgumentParser(); ap.add_argument("slug"); ap.add_argument("out"); ap.add_argument("--expect-min", type=int, default=150)
ap.add_argument("--grace-min", type=float, default=5, help="ignore terminal states this long (re-pushed kernels briefly report the previous version)")
a = ap.parse_args()
kag = str(Path(sys.executable).with_name("kaggle"))
prog = Progress(a.expect_min, f"kaggle {a.slug}", "min")
t0 = time.time(); state = ""
while True:
    r = subprocess.run([kag, "kernels", "status", a.slug], capture_output=True, text=True)
    state = (r.stdout + r.stderr).strip().split('"')[-2] if '"' in r.stdout else (r.stdout + r.stderr).strip()[-80:]
    el = (time.time() - t0) / 60
    prog.done = min(int(el), a.expect_min - 1); prog.update(0, desc=f"kaggle {a.slug}: {state}")
    if el >= a.grace_min and any(s in state for s in ("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(30)
Path(a.out).mkdir(parents=True, exist_ok=True)
subprocess.run([kag, "kernels", "output", a.slug, "-p", a.out])
prog.done = prog.total; prog.close("done")
print("final state:", state)
sys.exit(0 if "COMPLETE" in state else 1)
