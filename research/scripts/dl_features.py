"""Download SoccerNet per-game ResNet-PCA512 features (HF branch) for the caption games, and
SN-BAS-2025 valid/test. Progress in MB.
  python scripts/dl_features.py"""
import sys, glob, os, time, threading
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcs.progress import Progress
from SoccerNet.utils import getListGames
games = sorted({g for s in ("train", "valid", "test") for g in getListGames(s, task="caption")})
pats = [f"{g}/{h}_ResNET_TF2_PCA512.npy" for g in games for h in (1, 2)]
api = HfApi()
info = api.dataset_info("SoccerNet/SN-Features", revision="resnet-tf2-pca512", files_metadata=True)
size = {s.rfilename: s.size or 0 for s in info.siblings}
bas = api.dataset_info("SoccerNet/SN-BAS-2025", files_metadata=True)
bas_size = {s.rfilename: s.size or 0 for s in bas.siblings}
total = sum(size.get(p, 0) for p in pats) + bas_size["valid.zip"] + bas_size["test.zip"]
prog = Progress(int(total / 2**20), "download features + BAS", "MB")
FEAT = Path.home() / "soccerNet/hf/features_pca512"; BAS = Path.home() / "soccerNet/hf/bas2025"
done = threading.Event()
def watch():
    while not done.is_set():
        b = sum(p.stat().st_size for d in (FEAT, BAS) if d.exists() for p in d.rglob("*") if p.is_file())
        prog.done = int(b / 2**20); prog.update(0); time.sleep(3)
threading.Thread(target=watch, daemon=True).start()
snapshot_download("SoccerNet/SN-Features", repo_type="dataset", revision="resnet-tf2-pca512",
                  allow_patterns=pats, local_dir=str(FEAT), max_workers=8)
for f in ("valid.zip", "test.zip"):
    hf_hub_download("SoccerNet/SN-BAS-2025", f, repo_type="dataset", local_dir=str(BAS))
done.set(); prog.done = prog.total; prog.close("done")
print("features:", len(list(FEAT.rglob("*.npy"))), "files; BAS:", [p.name for p in BAS.glob("*.zip")])
