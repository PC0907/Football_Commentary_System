"""SoccerNet-Calibration(-2023) dataset read directly from the zip archives.

Targets per image, for every marking class c (see pitch.ALL_CLASSES):
  present[c]      1 if the class is annotated in the image
  pts[c, :P]      annotated points in normalised image coords (x, y) in [0, 1]
  pmask[c, :P]    valid-point mask
  dense[c]        rasterised polyline mask at 1/4 of the input resolution (aux. target)

Geometry is kept in *isotropic* normalised coordinates downstream: X = x * W/H, Y = y,
so that distances in x and y have the same scale (see model/losses).

Horizontal-flip augmentation maps the pitch by x -> -x, which swaps left/right classes
(top/bottom stay) and swaps the goal posts accordingly (table checked against geometry).
"""
from __future__ import annotations

import io
import json
import zipfile
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pitch import ALL_CLASSES, LINES

C = len(ALL_CLASSES)
IDX = {n: i for i, n in enumerate(ALL_CLASSES)}
P_MAX = 48
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _flip_name(n: str) -> str:
    table = {
        "Goal left post left ": "Goal right post right", "Goal right post right": "Goal left post left ",
        "Goal left post right": "Goal right post left", "Goal right post left": "Goal left post right",
        "Goal left crossbar": "Goal right crossbar", "Goal right crossbar": "Goal left crossbar",
    }
    if n in table:
        return table[n]
    if "left" in n:
        return n.replace("left", "right")
    if "right" in n:
        return n.replace("right", "left")
    return n


FLIP = [IDX[_flip_name(n)] for n in ALL_CLASSES]
from .pitch import mirror_name as _mirror_name  # noqa: E402
MIRROR_IDX = [IDX[_mirror_name(n)] for n in ALL_CLASSES]     # central symmetry (x,y) -> (-x,-y)


def _check_flip():
    for n in LINES:
        a, b = LINES[n]; m = LINES[_flip_name(n)]
        A, B = a * [-1, 1, 1], b * [-1, 1, 1]
        d = min(np.abs(A - m[0]).max() + np.abs(B - m[1]).max(), np.abs(A - m[1]).max() + np.abs(B - m[0]).max())
        assert d < 1e-9, n


_check_flip()


class SNCalibZip(Dataset):
    def __init__(self, zip_path: str, size=(644, 364), augment: bool = False, limit: Optional[int] = None):
        self.zip_path = zip_path
        self.size = size
        self.augment = augment
        self._z = None
        with zipfile.ZipFile(zip_path) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".json") and n.split("/")[-1][:5].isdigit())
        self.items: List[str] = names[:limit] if limit else names

    def _zip(self):
        if self._z is None:                      # one handle per worker process
            self._z = zipfile.ZipFile(self.zip_path)
        return self._z

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        z = self._zip()
        jn = self.items[i]
        ann = json.loads(z.read(jn))
        img = cv2.imdecode(np.frombuffer(z.read(jn[:-5] + ".jpg"), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        W, H = self.size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        flip = self.augment and np.random.rand() < 0.5
        present = np.zeros(C, np.float32)
        pts = np.zeros((C, P_MAX, 2), np.float32)
        pmask = np.zeros((C, P_MAX), np.float32)
        for name, plist in ann.items():
            if name not in IDX or not plist:
                continue
            k = IDX[name]
            p = np.array([[q["x"], q["y"]] for q in plist], np.float32)[:P_MAX]
            if flip:
                p[:, 0] = 1.0 - p[:, 0]
                k = FLIP[k]
            present[k] = 1.0
            pts[k, :len(p)] = p
            pmask[k, :len(p)] = 1.0
        if flip:
            img = img[:, ::-1].copy()
        if self.augment:
            img = _color_jitter(img)
        dense = np.zeros((C, H // 4, W // 4), np.uint8)
        for k in np.where(present > 0)[0]:
            p = pts[k, pmask[k] > 0] * [W // 4, H // 4]
            if len(p) >= 2:
                cv2.polylines(dense[k], [p.astype(np.int32)], False, 1, 1)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return {
            "image": torch.from_numpy(x.transpose(2, 0, 1)),
            "present": torch.from_numpy(present),
            "pts": torch.from_numpy(pts),
            "pmask": torch.from_numpy(pmask),
            "dense": torch.from_numpy(dense.astype(np.float32)),
            "name": jn,
        }


def _color_jitter(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-6, 6)) % 180
    hsv[..., 1] *= np.random.uniform(0.7, 1.3)
    hsv[..., 2] *= np.random.uniform(0.7, 1.3)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if np.random.rand() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out
