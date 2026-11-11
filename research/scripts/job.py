#!/usr/bin/env python
"""Job runner: every experiment is launched, logged and monitored through this.

  python scripts/job.py run NAME [--wait] -- <command ...>   launch a job
  python scripts/job.py status [--all]                         progress bars + ETA
  python scripts/job.py watch                                  live view (Ctrl-C to quit)
  python scripts/job.py tail JOB_ID [-n 40]                    last lines of a job's log
  python scripts/job.py list                                   job registry

For each job it records, in logs/:
  jobs.jsonl              one JSON line per event (start / end) with the exact command,
                          working dir, start/end time, exit code, log path
  <job_id>.log            full stdout+stderr of the job
  progress/<job_id>.json  live progress written by fcs.progress.Progress
Without --wait the job is detached (survives the terminal). With --wait the runner
blocks until the job finishes (used when Claude needs a completion notification).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PROG = LOGS / "progress"
REG = LOGS / "jobs.jsonl"


def _now_iso(t=None):
    return dt.datetime.fromtimestamp(t or time.time()).strftime("%Y-%m-%d %H:%M:%S")


def _register(ev: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(REG, "a") as f:
        f.write(json.dumps(ev) + "\n")


def _jobs() -> dict:
    jobs = {}
    if REG.exists():
        for line in REG.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            jobs.setdefault(e["job_id"], {}).update(e)
    return jobs


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def cmd_run(a):
    cmd = a.command
    if not cmd:
        sys.exit("no command given (use: job.py run NAME [--wait] -- <command>)")
    jid = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{a.name}"
    log = LOGS / f"{jid}.log"
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FCS_JOB_ID": jid, "PYTHONUNBUFFERED": "1"}
    if a.wait:
        _register(dict(job_id=jid, event="start", name=a.name, cmd=cmd, cwd=str(Path.cwd()),
                       start=_now_iso(), log=str(log), pid=os.getpid()))
        rc = _call(cmd, log, env)
        _finish(jid, rc)
        print(f"[{jid}] exit {rc}  log: {log.relative_to(ROOT)}")
        sys.exit(rc)
    # detached: re-invoke ourselves with --wait in a new session
    p = subprocess.Popen([sys.executable, __file__, "_wrap", jid, "--", *cmd],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    _register(dict(job_id=jid, event="start", name=a.name, cmd=cmd, cwd=str(Path.cwd()),
                   start=_now_iso(), log=str(log), pid=p.pid))
    print(f"started {jid}  (pid {p.pid})  log: {log.relative_to(ROOT)}")


def cmd_wrap(a):
    jid = a.job_id
    cmd = a.command
    log = LOGS / f"{jid}.log"
    _finish(jid, _call(cmd, log, os.environ))


def _call(cmd, log, env) -> int:
    with open(log, "w") as lf:
        try:
            return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        except OSError as e:          # command could not be started at all
            lf.write(f"job.py: cannot start {cmd!r}: {e}\n")
            return 127


def _finish(jid, rc):
    _register(dict(job_id=jid, event="end", end=_now_iso(), exit=rc, state="done" if rc == 0 else "failed"))
    p = PROG / f"{jid}.json"
    if p.exists():
        d = json.loads(p.read_text())
        d["state"] = "done" if rc == 0 else "failed"
        p.write_text(json.dumps(d))


def _bar(frac, width=28):
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + ("▌" if frac * width - full >= 0.5 else "") + " " * (width - full - (1 if frac * width - full >= 0.5 else 0))


def _fmt_s(s):
    if s is None:
        return "  --  "
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# Resource usage (task-manager style)
# ──────────────────────────────────────────────────────────────────────────────

def _nvsmi(query: str, kind: str = "gpu"):
    import shutil
    if not shutil.which("nvidia-smi"):
        return []
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    return [[c.strip() for c in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def _gpu_mem_by_pid():
    m = {}
    for row in _nvsmi("pid,used_memory", "apps"):
        try:
            m[int(row[0])] = m.get(int(row[0]), 0) + float(row[1])
        except (ValueError, IndexError):
            pass
    return m


def system_header() -> str:
    try:
        import psutil
    except ImportError:
        return "(install psutil for resource usage)"
    cpu = psutil.cpu_percent(interval=0.3)
    vm = psutil.virtual_memory(); sw = psutil.swap_memory()
    du = psutil.disk_usage(str(Path.home()))
    load = os.getloadavg()
    line = (f"CPU {cpu:5.1f}% ({psutil.cpu_count()} cores, load {load[0]:.1f})   "
            f"RAM {vm.used / 2**30:5.1f}/{vm.total / 2**30:.1f} GB ({vm.percent:.0f}%)   "
            f"swap {sw.used / 2**30:.1f} GB   disk free {du.free / 2**30:.0f} GB")
    for g in _nvsmi("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"):
        try:
            line += (f"\nGPU{g[0]} {g[1]}: util {g[2]}%  VRAM {float(g[3]) / 1024:.2f}/{float(g[4]) / 1024:.2f} GB  "
                     f"{g[5]}°C  {g[6]} W")
        except (ValueError, IndexError):
            pass
    return line


def _tree_usage(pid, gpu_by_pid):
    """CPU %, RSS (GB), GPU memory (GB) summed over a job's process tree."""
    try:
        import psutil
        root = psutil.Process(int(pid))
        procs = [root] + root.children(recursive=True)
    except Exception:
        return None
    for p in procs:
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(0.25)
    cpu = rss = 0.0
    gpu = 0.0
    for p in procs:
        try:
            cpu += p.cpu_percent(None)
            rss += p.memory_info().rss
            gpu += gpu_by_pid.get(p.pid, 0.0)
        except Exception:
            pass
    return cpu, rss / 2**30, gpu / 1024, len(procs), {p.pid for p in procs}


def render(show_all=False) -> str:
    jobs = _jobs()
    lines = []
    gpu_by_pid = _gpu_mem_by_pid()
    job_pids = set()
    now = time.time()
    for jid in sorted(jobs):
        j = jobs[jid]
        prog = PROG / f"{jid}.json"
        p = json.loads(prog.read_text()) if prog.exists() else None
        if "exit" in j:
            state = "done" if j["exit"] == 0 else f"FAILED({j['exit']})"
        else:
            state = "running" if _alive(j.get("pid")) else "dead?"
        if not show_all and state == "done":
            continue
        usage = ""
        if state == "running":
            u = _tree_usage(j.get("pid"), gpu_by_pid)
            if u:
                usage = f"  [CPU {u[0]:5.0f}%  RAM {u[1]:4.1f} GB  GPU {u[2]:4.2f} GB  procs {u[3]}]"
                job_pids |= u[4]
        if p and p["total"]:
            frac = p["done"] / p["total"]
            stage = f" [stage {p['stage']}/{p['n_stages']}]" if p.get("stage") else ""
            eta = p.get("eta_s") if state == "running" else 0
            fin = _now_iso(p["finish_at"]) if (state == "running" and p.get("finish_at")) else ("-" if state == "running" else j.get("end", "-"))
            lines.append(f"{jid:42s} {state:10s} |{_bar(frac)}| {100 * frac:5.1f}%  "
                         f"{p['done']}/{p['total']} {p['unit']}  {p['rate']:.2f} {p['unit']}/s  "
                         f"elapsed {_fmt_s(p['elapsed'])}  ETA {_fmt_s(eta)}  finish {fin}{usage}  {p['desc']}{stage}")
        else:
            t_start = time.mktime(time.strptime(j["start"], "%Y-%m-%d %H:%M:%S"))
            t_end = time.mktime(time.strptime(j["end"], "%Y-%m-%d %H:%M:%S")) if "end" in j else now
            el = t_end - t_start
            lines.append(f"{jid:42s} {state:10s} |{_bar(None)}|   (no progress reported)  elapsed {_fmt_s(el)}{usage}")
    # GPU processes not started through job.py (so nothing is hidden)
    other = []
    for pid, mem in gpu_by_pid.items():
        if pid not in job_pids:
            try:
                import psutil
                name = " ".join(psutil.Process(pid).cmdline())[:90]
            except Exception:
                name = "?"
            other.append(f"   pid {pid}: GPU {mem / 1024:.2f} GB  {name}")
    body = "\n".join(lines) if lines else "no running jobs (use --all to include finished ones)"
    if other:
        body += "\n\nother GPU processes (not started via job.py):\n" + "\n".join(other)
    return system_header() + "\n" + "─" * 100 + "\n" + body


def cmd_status(a):
    print(render(a.all))


def cmd_watch(a):
    try:
        while True:
            os.system("clear")
            print(f"jobs @ {_now_iso()}   (Ctrl-C to quit)\n")
            print(render(a.all))
            time.sleep(2)
    except KeyboardInterrupt:
        pass


def cmd_tail(a):
    jobs = _jobs()
    m = [j for j in jobs if a.job_id in j]
    if not m:
        sys.exit("no such job")
    log = Path(jobs[m[-1]]["log"])
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    print("\n".join(lines[-a.n:]))


def cmd_list(a):
    for jid, j in sorted(_jobs().items()):
        print(f"{jid:42s} {j.get('state', 'running'):8s} start {j['start']}  end {j.get('end', '-'):19s}  "
              f"{' '.join(j['cmd'])[:120]}")


def main():
    # split "<job.py options> -- <command>" explicitly (argparse.REMAINDER would swallow
    # options such as --wait that appear before the command)
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        k = argv.index("--")
        argv, command = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="c", required=True)
    r = sub.add_parser("run"); r.add_argument("name"); r.add_argument("--wait", action="store_true")
    w = sub.add_parser("_wrap"); w.add_argument("job_id")
    s = sub.add_parser("status"); s.add_argument("--all", action="store_true")
    wa = sub.add_parser("watch"); wa.add_argument("--all", action="store_true")
    t = sub.add_parser("tail"); t.add_argument("job_id"); t.add_argument("-n", type=int, default=40)
    sub.add_parser("list")
    a = ap.parse_args(argv)
    a.command = command
    {"run": cmd_run, "_wrap": cmd_wrap, "status": cmd_status, "watch": cmd_watch,
     "tail": cmd_tail, "list": cmd_list}[a.c](a)


if __name__ == "__main__":
    main()
