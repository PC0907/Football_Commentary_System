"""Progress reporting for long jobs.

Every long-running script reports progress through ``Progress``; it writes a small JSON
status file (throttled to ~1 write/s) that ``scripts/job.py status|watch`` renders as a
progress bar with rate and estimated finish time. Nothing is printed to the console
except an occasional one-line summary, so job logs stay small.

    from fcs.progress import Progress
    with Progress(total=len(items), desc="tracking T2", unit="seq") as p:
        for x in items:
            ...
            p.update(1)

The status file is ``logs/progress/<job_id>.json`` where job_id comes from the
FCS_JOB_ID environment variable (set by job.py) or defaults to the script name.
A job can have several stages; each ``Progress`` instance is one stage.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs"
PROG_DIR = LOG_ROOT / "progress"


def job_id() -> str:
    jid = os.environ.get("FCS_JOB_ID")
    if jid:
        return jid
    return Path(sys.argv[0]).stem or "interactive"


class Progress:
    def __init__(self, total: int, desc: str = "", unit: str = "it", stage: Optional[int] = None,
                 n_stages: Optional[int] = None):
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.done = 0
        self.t0 = time.time()
        self._last = 0.0
        self.stage = stage
        self.n_stages = n_stages
        PROG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = PROG_DIR / f"{job_id()}.json"
        self._write("running", force=True)

    # ------------------------------------------------------------------
    def update(self, n: int = 1, desc: Optional[str] = None):
        self.done += n
        if desc:
            self.desc = desc
        self._write("running")

    def _write(self, state: str, force: bool = False):
        now = time.time()
        if not force and now - self._last < 1.0 and self.done < self.total:
            return
        self._last = now
        el = now - self.t0
        rate = self.done / el if el > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        d = dict(job=job_id(), desc=self.desc, unit=self.unit, done=self.done, total=self.total,
                 started=self.t0, updated=now, elapsed=el, rate=rate,
                 eta_s=eta, finish_at=(now + eta) if eta is not None else None,
                 state=state, stage=self.stage, n_stages=self.n_stages, pid=os.getpid())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(self.path)

    def close(self, state: str = "done"):
        self._write(state, force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close("failed" if exc_type else "stage-done")
        return False


def track(it: Iterable, total: Optional[int] = None, desc: str = "", unit: str = "it"):
    total = total if total is not None else len(it)  # type: ignore[arg-type]
    with Progress(total, desc, unit) as p:
        for x in it:
            yield x
            p.update(1)
