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
STEPS = {"cpu": 2, "gpu": 4, "kaggle": 9}
prog = Progress(sum(STEPS.values()), "overnight pipeline v2", "step")
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
    pass                                     # test-set oracle tracking + ball already done (overnight v1)


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


def kaggle_status(slug):
    r = subprocess.run([str(Path(PY).with_name("kaggle")), "kernels", "status", slug], capture_output=True, text=True)
    return r.stdout + r.stderr


def push(task, name, args):
    rc = subprocess.call([PY, "kaggle/push_kernel.py", task, name, args], cwd=ROOT)
    with LOCK:
        status[f"push-{name}"] = dict(rc=rc); prog.update(1, desc=f"overnight — pushed {name}")
    return rc


def detector_and_tracking(name, expect_min, tag):
    out = ROOT / "outputs" / "detector" / tag
    step("kaggle", f"kaggle-{tag}", [PY, "scripts/kaggle_watch.py", f"gk8247/{name}", str(out), "--expect-min", str(expect_min)])
    dets = list(out.rglob("dets/test/*.txt"))
    cache = ROOT / "cache" / "dets" / tag / "test"
    cache.mkdir(parents=True, exist_ok=True)
    for d in dets:
        (cache / d.name).write_text(d.read_text())
    if len(dets) >= 49:
        step("kaggle", f"track-test-{tag}", [PY, "scripts/run_tracking.py", "--split", "test", "--det", tag, "--workers", "4",
                                             "--variants", "T0-fixed", "T1", "T2", "T3", "T4"])
        step("kaggle", f"ball-test-{tag}", [PY, "scripts/run_ball.py", "--split", "test", "--det", tag,
                                            "--variants", "B0", "B1", "--calib", "fusedL", "--workers", "4"])
    else:
        with LOCK:
            status[f"track-test-{tag}"] = dict(rc=-1, note=f"only {len(dets)} detection files"); prog.update(2)


def kaggle_stream():
    # slot 1: frees when PitchGeo training finishes -> YOLO26m detector
    wait_until(lambda: job_finished("kaggle-pitchgeo-v2"))
    push("detector", "fcs-det-yolo26m", "--arch yolo26m --epochs 40 --imgsz 1280 --batch 8")
    t1 = threading.Thread(target=detector_and_tracking, args=("fcs-det-yolo26m", 300, "yolo26m_snmot"))
    t1.start()
    # slot 2: frees when the commentary-4B run finishes -> RF-DETR-M detector
    step("kaggle", "kaggle-commentary-4b", [PY, "scripts/kaggle_watch.py", "gk8247/fcs-commentary-g2-4b",
                                            "outputs/commentary/kaggle_g2_4b", "--expect-min", "240"])
    push("detector", "fcs-det-rfdetr-m", "--arch rfdetr-m --epochs 30 --imgsz 1120 --batch 8")
    t2 = threading.Thread(target=detector_and_tracking, args=("fcs-det-rfdetr-m", 360, "rfdetr_m_snmot"))
    t2.start()
    t1.join()
    # first slot to free again -> PitchGeo deterministic ablation
    push("pitchgeo", "fcs-pitchgeo-deterministic", "--epochs 25 --bs 16 --workers 4 --deterministic")
    time.sleep(120)
    step("kaggle", "kaggle-pitchgeo-det", [PY, "scripts/kaggle_watch.py", "gk8247/fcs-pitchgeo-deterministic",
                                           "outputs/pitchgeo/kaggle_det", "--expect-min", "240"])
    t2.join()


if __name__ == "__main__":
    th = [threading.Thread(target=f) for f in (cpu_stream, gpu_stream, kaggle_stream)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    prog.close("done")
    print(json.dumps(status, indent=1))
