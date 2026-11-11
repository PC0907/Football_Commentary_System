"""3-D pitch model in SoccerNet-calibration conventions (written from the published
dimensions; class names match SN-Calibration annotations).

Frame: origin at the centre mark, x along the length (-52.5 .. 52.5, "left" goal at -x),
y across (-34 = "top" = far touchline, +34 = "bottom" = near touchline), z into the ground
(goal crossbar at z = -2.44).

Marking types:
  LINE   — straight 3-D segment (17 ground lines + 6 goal elements)
  CIRCLE — circle (or arc) on the ground plane
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

L, W = 105.0, 68.0
PA_L, PA_W = 16.5, 40.32       # penalty area
GA_L, GA_W = 5.5, 18.32        # goal area
GOAL_W, GOAL_H = 7.32, 2.44
R_CIRCLE = 9.15
PENALTY_MARK = 11.0

_h, _w = L / 2, W / 2


def _p(x, y, z=0.0):
    return np.array([x, y, z], float)


LINES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "Side line top": (_p(-_h, -_w), _p(_h, -_w)),
    "Side line bottom": (_p(-_h, _w), _p(_h, _w)),
    "Side line left": (_p(-_h, -_w), _p(-_h, _w)),
    "Side line right": (_p(_h, -_w), _p(_h, _w)),
    "Middle line": (_p(0, -_w), _p(0, _w)),
    "Big rect. left top": (_p(-_h, -PA_W / 2), _p(-_h + PA_L, -PA_W / 2)),
    "Big rect. left bottom": (_p(-_h, PA_W / 2), _p(-_h + PA_L, PA_W / 2)),
    "Big rect. left main": (_p(-_h + PA_L, -PA_W / 2), _p(-_h + PA_L, PA_W / 2)),
    "Big rect. right top": (_p(_h, -PA_W / 2), _p(_h - PA_L, -PA_W / 2)),
    "Big rect. right bottom": (_p(_h, PA_W / 2), _p(_h - PA_L, PA_W / 2)),
    "Big rect. right main": (_p(_h - PA_L, -PA_W / 2), _p(_h - PA_L, PA_W / 2)),
    "Small rect. left top": (_p(-_h, -GA_W / 2), _p(-_h + GA_L, -GA_W / 2)),
    "Small rect. left bottom": (_p(-_h, GA_W / 2), _p(-_h + GA_L, GA_W / 2)),
    "Small rect. left main": (_p(-_h + GA_L, -GA_W / 2), _p(-_h + GA_L, GA_W / 2)),
    "Small rect. right top": (_p(_h, -GA_W / 2), _p(_h - GA_L, -GA_W / 2)),
    "Small rect. right bottom": (_p(_h, GA_W / 2), _p(_h - GA_L, GA_W / 2)),
    "Small rect. right main": (_p(_h - GA_L, -GA_W / 2), _p(_h - GA_L, GA_W / 2)),
    # goal elements (off the ground plane) — left goal at -x, right at +x.
    # SN naming: "post left/right" as seen from the pitch centre looking at the goal.
    "Goal left crossbar": (_p(-_h, -GOAL_W / 2, -GOAL_H), _p(-_h, GOAL_W / 2, -GOAL_H)),
    "Goal left post left ": (_p(-_h, GOAL_W / 2, 0), _p(-_h, GOAL_W / 2, -GOAL_H)),
    "Goal left post right": (_p(-_h, -GOAL_W / 2, 0), _p(-_h, -GOAL_W / 2, -GOAL_H)),
    "Goal right crossbar": (_p(_h, -GOAL_W / 2, -GOAL_H), _p(_h, GOAL_W / 2, -GOAL_H)),
    "Goal right post left": (_p(_h, -GOAL_W / 2, 0), _p(_h, -GOAL_W / 2, -GOAL_H)),
    "Goal right post right": (_p(_h, GOAL_W / 2, 0), _p(_h, GOAL_W / 2, -GOAL_H)),
}

# circles: (centre, radius, angular range of the drawn part)
_arc = np.arccos((PA_L - PENALTY_MARK) / R_CIRCLE)     # half-angle of the "D" outside the box
CIRCLES: Dict[str, Tuple[np.ndarray, float, Tuple[float, float]]] = {
    "Circle central": (_p(0, 0), R_CIRCLE, (-np.pi, np.pi)),
    "Circle left": (_p(-_h + PENALTY_MARK, 0), R_CIRCLE, (-_arc, _arc)),
    "Circle right": (_p(_h - PENALTY_MARK, 0), R_CIRCLE, (np.pi - _arc, np.pi + _arc)),
}

LINE_CLASSES: List[str] = list(LINES)
CIRCLE_CLASSES: List[str] = list(CIRCLES)
ALL_CLASSES: List[str] = LINE_CLASSES + CIRCLE_CLASSES
GROUND_LINES = [c for c in LINE_CLASSES if not c.startswith("Goal")]

# Central symmetry of the pitch (x -> -x, y -> -y). SN's metric scores a prediction against
# both the GT labels and their mirrored labels and keeps the better one.
MIRROR = {
    "Side line top": "Side line bottom", "Side line bottom": "Side line top",
    "Side line left": "Side line right", "Side line right": "Side line left",
    "Middle line": "Middle line",
    "Big rect. left top": "Big rect. right bottom", "Big rect. left bottom": "Big rect. right top",
    "Big rect. left main": "Big rect. right main", "Big rect. right top": "Big rect. left bottom",
    "Big rect. right bottom": "Big rect. left top", "Big rect. right main": "Big rect. left main",
    "Small rect. left top": "Small rect. right bottom", "Small rect. left bottom": "Small rect. right top",
    "Small rect. left main": "Small rect. right main", "Small rect. right top": "Small rect. left bottom",
    "Small rect. right bottom": "Small rect. left top", "Small rect. right main": "Small rect. left main",
    "Circle left": "Circle right", "Circle central": "Circle central", "Circle right": "Circle left",
    "Goal left crossbar": "Goal right crossbar", "Goal right crossbar": "Goal left crossbar",
    "Goal left post left ": "Goal right post left", "Goal right post left": "Goal left post left ",
    "Goal left post right": "Goal right post right", "Goal right post right": "Goal left post right",
}


def mirror_name(name: str) -> str:
    return MIRROR.get(name, name)


def sample_polyline(name: str, n: int = 30) -> np.ndarray:
    """3-D points along a marking."""
    if name in LINES:
        a, b = LINES[name]
        t = np.linspace(0, 1, n)[:, None]
        return a + t * (b - a)
    c, r, (t0, t1) = CIRCLES[name]
    t = np.linspace(t0, t1, n)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t), np.zeros(n)]


def to_topleft(xy: np.ndarray) -> np.ndarray:
    """SN centre-origin pitch coords -> tracking code's top-left origin (0..105, 0..68)."""
    return np.asarray(xy)[..., :2] + np.array([_h, _w])
