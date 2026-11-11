"""Download one file from a Hugging Face dataset repo with progress reporting.
  python scripts/dl_hf_file.py REPO FILE LOCAL_DIR"""
import sys, time, threading
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcs.progress import Progress
repo, fname, local = sys.argv[1], sys.argv[2], Path(sys.argv[3])
size = next(s.size for s in HfApi().dataset_info(repo, files_metadata=True).siblings if s.rfilename == fname)
prog = Progress(int(size / 2**20), f"download {repo}/{fname}", "MB")
done = threading.Event()
def watch():
    while not done.is_set():
        part = sum(p.stat().st_size for p in local.rglob("*.incomplete")) if local.exists() else 0
        prog.done = int(part / 2**20); prog.update(0); time.sleep(2)
threading.Thread(target=watch, daemon=True).start()
p = hf_hub_download(repo, fname, repo_type="dataset", local_dir=str(local))
done.set(); prog.done = prog.total; prog.close("done"); print(p)
