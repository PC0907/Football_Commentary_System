"""Overnight orchestrator: three parallel streams, every step a logged job (scripts/job.py).

  python scripts/job.py run overnight -- python scripts/overnight.py

Streams
  cpu    : final TEST-set tracking (persons: oracle boxes + repo detector) and ball tracking
  gpu    : wait for local QLoRA 2B -> score G2-2B on test -> G1 zero-shot on test ->
           wait for the PitchGeo checkpoint -> evaluate it on SN-Calib valid + test
  kaggle : wait for PitchGeo training -> push the deterministic ablation -> watch it and the
           commentary-4B kernel, download outputs
Progress of this orchestrator = finished steps / total steps; each step shows its own bar.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fcs.progress import Progress  # noqa: E402

PY = sys.executable
JOB = [PY, str(ROOT / "scripts" / "job.py"), "run"]
REG = ROOT / "logs" / "jobs.jsonl"
CALIB = Path.home() / "soccerNet/hf/calib2023"
LOCK = threading.Lock()
STEPS = {"cpu": 2, "gpu": 4, "kaggle": 4}
prog = Progress(sum(STEPS.values()), "overnight pipeline", "step")
status = {}


def step(stream, name, cmd, env=None):
    t0 = time.time()
    rc = subprocess.call(JOB + [name, "--wait", "--"] + cmd, cwd=ROOT, env=env)
    with LOCK:
        status[name] = dict(rc=rc, minutes=round((time.time() - t0) / 60, 1))
        prog.update(1, desc=f"overnight pipeline — last: {name} (rc {rc})")
        (ROOT / "logs" / "overnight_status.json").write_text(json.dumps(status, indent=1))
    return rc


def job_finished(name_part):
    if not REG.exists():
        return False
    ends = {}
    for line in REG.read_text().splitlines():
        e = json.loads(line)
        if name_part in e["job_id"]:
            ends[e["job_id"]] = e.get("event") == "end" or ends.get(e["job_id"], False)
    return bool(ends) and all(ends.values())


def wait_until(pred, poll=60):
    while not pred():
        time.sleep(poll)


def cpu_stream():
    step("cpu", "track-test-oracle", [PY, "scripts/run_tracking.py", "--split", "test", "--det", "oracle", "--workers", "4",
                                       "--variants", "T0", "T0-fixed", "T1-nocmc", "T1", "T2-allstage", "T2", "T2-fusedL",
                                       "T2-isoR", "T2-novmax", "T3", "T4-noapp", "T4"])
    step("cpu", "ball-test", [PY, "scripts/run_ball.py", "--split", "test", "--det", "yolov8m_repo",
                              "--variants", "B0", "B1", "B2", "B3", "--calib", "fusedL", "--workers", "4"])


def gpu_stream():
    import os
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    wait_until(lambda: job_finished("lora-g2-qwen35-2b"))
    step("gpu", "g2-2b-test", [PY, "scripts/commentary_llm.py", "--split", "test", "--n", "0", "--model", "Qwen/Qwen3.5-2B",
                               "--adapter", "outputs/commentary/lora_g2_2b/adapter", "--tag", "g2_2b", "--bs", "16"], env)
    step("gpu", "g1-2b-test", [PY, "scripts/commentary_llm.py", "--split", "test", "--n", "0", "--model", "Qwen/Qwen3.5-2B",
                               "--tag", "g1_2b", "--bs", "16"], env)
    ck = ROOT / "outputs/pitchgeo/kaggle_run1/run/best.pt"
    wait_until(lambda: ck.exists() or job_finished("kaggle-pitchgeo-v2"))
    if ck.exists():
        for split in ("valid", "test"):
            step("gpu", f"pitchgeo-eval-{split}", [PY, "scripts/eval_pitchgeo.py", "--ckpt", str(ck),
                                                    "--zip", str(CALIB / f"{split}.zip"), "--tag", f"K3_prob_{split}"])
    else:
        with LOCK:
            status["pitchgeo-eval"] = dict(rc=-1, note="no checkpoint (kaggle run failed?)")
            prog.update(2)


def kaggle_stream():
    wait_until(lambda: job_finished("kaggle-pitchgeo-v2"))
    with LOCK:
        prog.update(1, desc="overnight pipeline — PitchGeo kaggle run finished")
    rc = subprocess.call([PY, "kaggle/push_kernel.py", "pitchgeo", "fcs-pitchgeo-deterministic",
                          "--epochs 25 --bs 16 --workers 4 --deterministic"], cwd=ROOT)
    with LOCK:
        status["push-pitchgeo-deterministic"] = dict(rc=rc); prog.update(1)
    t1 = threading.Thread(target=step, args=("kaggle", "kaggle-commentary-4b",
                          [PY, "scripts/kaggle_watch.py", "gk8247/fcs-commentary-g2-4b", "outputs/commentary/kaggle_g2_4b",
                           "--expect-min", "240"]))
    t1.start()
    time.sleep(120)
    step("kaggle", "kaggle-pitchgeo-det", [PY, "scripts/kaggle_watch.py", "gk8247/fcs-pitchgeo-deterministic",
                                           "outputs/pitchgeo/kaggle_det", "--expect-min", "180"])
    t1.join()


if __name__ == "__main__":
    th = [threading.Thread(target=f) for f in (cpu_stream, gpu_stream, kaggle_stream)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    prog.close("done")
    print(json.dumps(status, indent=1))
