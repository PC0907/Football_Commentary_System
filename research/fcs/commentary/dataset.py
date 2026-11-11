"""Build (game state -> commentary) examples from SoccerNet-Caption / MatchTime labels.

Sources
  captions : MatchTime aligned labels (train/valid) and SN-Caption-test-align (test;
             only captions with a manual `gt_gameTime`, = the 3,267 of the published protocol)
  actions  : SoccerNet Labels-v2 (17 action classes with team side), human annotations.

Game state at caption time t (only information knowable at t — no leakage):
  teams, half and clock, running score and cards (from Labels-v2 up to t), the actions in
  the window [t - before, t + after] with their offsets and sides. The caption's own
  `label` field (e.g. 'y-card', 'corner') is NOT used: it is derived from the target.

Each example: dict(game, split, half, t, state (structured dict), target (anonymised),
target_identified, important).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

MT_ROOT = Path(os.environ.get("MATCHTIME_ROOT", Path.home() / "soccerNet/matchtime_hf"))
SN_LABELS = Path(os.environ.get("SN_LABELS_ROOT", Path.home() / "soccerNet/hf/labels"))

SCORING = {"Goal"}
CARDS = {"Yellow card", "Red card", "Yellow->red card"}


def _gt_seconds(gt: str) -> tuple:
    half, mmss = gt.split(" - ")
    m, s = mmss.strip().split(":")
    return int(half), int(m) * 60 + int(s)


def _sn_game_dir(mt_rel: str) -> Path:
    """'england_epl_2015-2016/<game>' -> SN-Labels 'england_epl/2015-2016/<game>'."""
    league_season, game = mt_rel.split("/", 1)
    league, season = league_season.rsplit("_", 1)
    return SN_LABELS / league / season / game


def load_actions(game_dir: Path) -> List[dict]:
    p = game_dir / "Labels-v2.json"
    if not p.exists():
        return []
    out = []
    for a in json.loads(p.read_text())["annotations"]:
        half, sec = _gt_seconds(a["gameTime"])
        out.append(dict(half=half, t=sec, label=a["label"], team=a.get("team", "not applicable"),
                        visibility=a.get("visibility", "")))
    return sorted(out, key=lambda a: (a["half"], a["t"]))


def game_state(actions: List[dict], half: int, t: int, home: str, away: str,
               before: int = 25, after: int = 5) -> dict:
    past = [a for a in actions if (a["half"], a["t"]) <= (half, t)]
    score = {"home": 0, "away": 0}
    cards = {"home": 0, "away": 0}
    for a in past:
        if a["label"] in SCORING and a["team"] in score:
            score[a["team"]] += 1
        if a["label"] in CARDS and a["team"] in cards:
            cards[a["team"]] += 1
    window = [dict(dt=a["t"] - t, label=a["label"], team=a["team"]) for a in actions
              if a["half"] == half and -before <= a["t"] - t <= after]
    return dict(home=home, away=away, half=half, clock=f"{t // 60:02d}:{t % 60:02d}",
                score=score, cards=cards, events=window)


def build_split(split: str, before: int = 60, after: int = 20) -> List[dict]:
    if split == "test":
        files = glob.glob(str(MT_ROOT / "SN-Caption-test-align" / "**" / "*.json"), recursive=True)
        base = MT_ROOT / "SN-Caption-test-align"
    else:
        files = glob.glob(str(MT_ROOT / "MatchTime" / split / "**" / "*.json"), recursive=True)
        base = MT_ROOT / "MatchTime" / split
    out = []
    for f in sorted(files):
        rel = str(Path(f).parent.relative_to(base))
        d = json.loads(Path(f).read_text())
        actions = load_actions(_sn_game_dir(rel))
        home, away = d.get("gameHomeTeam", "home"), d.get("gameAwayTeam", "away")
        for a in d["annotations"]:
            gt = a.get("gt_gameTime", "").strip() if split == "test" else a.get("gameTime", "").strip()
            if not gt:
                continue
            half, t = _gt_seconds(gt)
            out.append(dict(game=rel, split=split, half=half, t=t,
                            state=game_state(actions, half, t, home, away, before, after),
                            target=a["anonymized"], target_identified=a.get("identified", ""),
                            important=str(a.get("important", "")).lower() == "true",
                            has_actions=bool(actions)))
    return out


def state_to_text(s: dict, anonymise_teams: bool = True) -> str:
    """Compact, deterministic text rendering of the game state (model input)."""
    H = "[TEAM_HOME]" if anonymise_teams else s["home"]
    A = "[TEAM_AWAY]" if anonymise_teams else s["away"]
    side = {"home": H, "away": A}
    ev = "; ".join(f"{e['dt']:+d}s {e['label']}" + (f" ({side[e['team']]})" if e["team"] in side else "")
                   for e in s["events"]) or "none"
    return (f"Half {s['half']}, {s['clock']}. Score {H} {s['score']['home']}-{s['score']['away']} {A}. "
            f"Cards {H} {s['cards']['home']}, {A} {s['cards']['away']}. Actions around now: {ev}.")
