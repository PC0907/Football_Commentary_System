"""SoccerNet-Tracking (SNMOT) I/O."""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SNMOT_ROOT = Path(os.environ.get("SNMOT_ROOT", "/home/fawwaz/soccerNet/hf/tracking"))

# role strings in gameinfo.ini -> coarse class used by our trackers
ROLE_PLAYER, ROLE_GK, ROLE_REF, ROLE_BALL, ROLE_OTHER = 0, 1, 2, 3, 4


@dataclass
class Sequence:
    name: str
    root: Path
    fps: float
    length: int
    width: int
    height: int
    game_id: int
    action_class: str
    game_time_start: str
    tracklet_role: Dict[int, str] = field(default_factory=dict)   # gt id -> "player team left;4"

    def img_path(self, frame: int) -> Path:
        return self.root / "img1" / f"{frame:06d}.jpg"

    # ---------------------------------------------------------------- loaders
    def _read_mot(self, rel: str) -> np.ndarray:
        p = self.root / rel
        if not p.exists():
            return np.zeros((0, 10))
        a = np.loadtxt(p, delimiter=",", ndmin=2)
        return a

    def gt(self) -> np.ndarray:
        """(N, 10) MOT rows: frame, id, x, y, w, h, conf, cls, vis, -"""
        return self._read_mot("gt/gt.txt")

    def det(self) -> np.ndarray:
        return self._read_mot("det/det.txt")

    def role_of(self, gt_id: int) -> int:
        r = self.tracklet_role.get(int(gt_id), "").strip().lower()
        if r.startswith("ball"):
            return ROLE_BALL
        if r.startswith("goalkeeper"):
            return ROLE_GK
        if r.startswith("referee"):
            return ROLE_REF
        if r.startswith("player"):
            return ROLE_PLAYER
        return ROLE_OTHER

    def team_of(self, gt_id: int) -> Optional[str]:
        r = self.tracklet_role.get(int(gt_id), "").lower()
        if "team left" in r:
            return "left"
        if "team right" in r:
            return "right"
        return None


def load_sequence(path: Path) -> Sequence:
    path = Path(path)
    si = configparser.ConfigParser()
    si.read(path / "seqinfo.ini")
    s = si["Sequence"]
    gi = configparser.ConfigParser()
    gi.read(path / "gameinfo.ini")
    g = gi["Sequence"]
    roles = {}
    for k, v in g.items():
        if k.startswith("trackletid_"):
            roles[int(k.split("_")[1])] = v
    return Sequence(
        name=s["name"], root=path, fps=float(s["frameRate"]), length=int(s["seqLength"]),
        width=int(s["imWidth"]), height=int(s["imHeight"]),
        game_id=int(g.get("gameID", -1)), action_class=g.get("actionClass", ""),
        game_time_start=g.get("gameTimeStart", ""), tracklet_role=roles,
    )


def list_sequences(split: str = "test") -> List[Sequence]:
    d = SNMOT_ROOT / split
    return [load_sequence(p) for p in sorted(d.iterdir()) if p.is_dir()]


def detections_by_frame(rows: np.ndarray, length: int) -> List[np.ndarray]:
    """Group MOT rows into per-frame arrays [x, y, w, h, score, cls]."""
    out: List[np.ndarray] = [np.zeros((0, 6)) for _ in range(length + 1)]
    if len(rows) == 0:
        return out
    frames = rows[:, 0].astype(int)
    for f in np.unique(frames):
        r = rows[frames == f]
        cls = r[:, 7] if r.shape[1] > 7 else -np.ones(len(r))
        out[f] = np.column_stack([r[:, 2:6], r[:, 6], cls])
    return out


def oracle_detections(seq: Sequence, with_class: bool = True) -> List[np.ndarray]:
    """GT boxes with identities removed. The class column is filled from gameinfo
    roles when ``with_class`` (a detector would also output a class)."""
    gt = seq.gt()
    rows = gt.copy()
    rows[:, 6] = 1.0
    if with_class:
        rows[:, 7] = [seq.role_of(i) for i in gt[:, 1]]
    rows[:, 1] = -1
    return detections_by_frame(rows, seq.length)


def write_mot(path: Path, rows: List[tuple]) -> None:
    """rows: (frame, id, x, y, w, h, score)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    with open(path, "w") as f:
        for fr, tid, x, y, w, h, s in rows:
            f.write(f"{int(fr)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.3f},-1,-1,-1\n")
