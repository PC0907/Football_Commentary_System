"""
Event detection abstraction — multiple backends behind one interface.

Architecture
------------
BaseEventDetector  (ABC)
├── RuleBasedEventDetector   wraps FootballEventDetector from 2D_event_detector.py
│                            Converts pipeline field_positions → FootballEventDetector
│                            input format and normalises output to the canonical schema.
├── MLEventDetector          Stub for future action-recognition models (C3D, SlowFast,
│                            VideoSwin, etc.).  Raises NotImplementedError with guidance.
└── EnsembleEventDetector    Combines any number of detectors and deduplicates events.

EventDetectorFactory.create(config: dict) → BaseEventDetector
  Supports registration of custom detector classes at runtime.

Canonical event schema
----------------------
Every detector must return events in this form:
{
    'type'               : str,         # 'shot' | 'pass' | 'goal' | 'corner_kick'
                                        #   | 'free_kick' | 'foul' | 'penalty'
    'timestamp'          : float,       # seconds from start of video
    'frame'              : int,
    'team'               : str,         # 'team_a' | 'team_b' | ''
    'player_id'          : str | int,   # primary actor (track_id), or ''
    'secondary_player_id': str | int,   # pass recipient, or ''
    'confidence'         : float,       # 0-1
    'metadata'           : dict,        # type-specific extras
}
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .football import FootballEventDetector

import numpy as np

log = logging.getLogger(__name__)

# ── Object-id constants (must match detection_utils.LABELS) ───────────────────
_OID_PLAYER_L = 0   # Team-A outfield
_OID_PLAYER_R = 1   # Team-B outfield
_OID_GK_L     = 2   # Team-A goalkeeper
_OID_GK_R     = 3   # Team-B goalkeeper
_OID_BALL     = 4


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class BaseEventDetector(ABC):
    """
    Common interface for all event detectors.

    ``detect(frame_snapshots)`` is the only required method.  Each snapshot in
    the list is a dict produced by VideoProcessor:

        {
            'timestamp'      : float,
            'tracking'       : list[dict],   # tracker output (pixel coords)
            'field_positions': list[dict],   # homography output (world metres)
            'fps'            : float,
        }

    Returns a list of event dicts normalised to the canonical schema above.
    """

    @abstractmethod
    def detect(self, frame_snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...

    def set_fps(self, fps: float) -> None:
        """Optional: update the detector's fps after the video is opened."""


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based detector  (wraps FootballEventDetector)
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedEventDetector(BaseEventDetector):
    """
    Physics / geometry rule-based event detection.

    Wraps ``FootballEventDetector`` from ``2D_event_detector.py`` (loaded via
    importlib to avoid the leading-digit import restriction).

    Pipeline adaptation
    -------------------
    field_positions  (homography output):
        [{'object_id': 0, 'track_id': 5, 'world_x_meters': 23.5,
          'world_y_meters': 34.1}, ...]

    FootballEventDetector.process_frame needs:
        ball_pos    : (x, y) metres — from object_id == 4
        players_pos : {'home': {track_id: (x, y)},
                       'away': {track_id: (x, y)}}

    Output normalisation
    --------------------
    FootballEventDetector uses team labels 'home' / 'away'.
    We normalise to 'team_a' / 'team_b'.
    """

    # How often (in frames) to actually run detection on the latest snapshot.
    # Running every frame is redundant — the rule-based detector checks state
    # changes that accumulate across the buffer anyway.
    _DETECT_EVERY = 1

    def __init__(self, fps: float = 30.0, **kwargs):
        self._fps  = fps
        self._kwargs = kwargs
        self._inner = self._build_inner(fps, **kwargs)
        self._frame_count = 0

    def _build_inner(self, fps, **kwargs):
        """Instantiate FootballEventDetector from pipeline.events.football."""
        return FootballEventDetector(fps=fps, **kwargs)

    def set_fps(self, fps: float) -> None:
        """Rebuild inner detector with the real fps from the opened video."""
        if abs(fps - self._fps) > 1.0:
            log.info("RuleBasedEventDetector: updating fps %.1f → %.1f", self._fps, fps)
            self._fps  = fps
            self._inner = self._build_inner(fps, **self._kwargs)

    def detect(self, frame_snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not frame_snapshots:
            return []

        # Process only the most recent snapshot to avoid duplicate event firing
        snap = frame_snapshots[-1]
        self._frame_count += 1

        field_positions: List[Dict] = snap.get("field_positions", [])
        timestamp: float = snap.get("timestamp", 0.0)

        ball_pos    = self._extract_ball(field_positions)
        players_pos = self._extract_players(field_positions)

        if ball_pos is None:
            return []

        try:
            raw_events = self._inner.process_frame(
                self._frame_count, ball_pos, players_pos
            )
        except Exception as exc:
            log.warning("FootballEventDetector.process_frame failed: %s", exc)
            return []

        return [self._normalise(e, timestamp, self._frame_count) for e in raw_events]

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_ball(
        field_positions: List[Dict],
    ) -> Optional[Tuple[float, float]]:
        for fp in field_positions:
            if fp.get("object_id") == _OID_BALL:
                return (
                    float(fp.get("world_x_meters", 0)),
                    float(fp.get("world_y_meters", 0)),
                )
        return None

    @staticmethod
    def _extract_players(
        field_positions: List[Dict],
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        players: Dict[str, Dict] = {"home": {}, "away": {}}
        for fp in field_positions:
            oid = fp.get("object_id", -1)
            tid = str(fp.get("object_id", fp.get("track_id", "?")))
            x   = float(fp.get("world_x_meters", 0))
            y   = float(fp.get("world_y_meters", 0))
            if oid in (_OID_PLAYER_L, _OID_GK_L):
                players["home"][tid] = (x, y)
            elif oid in (_OID_PLAYER_R, _OID_GK_R):
                players["away"][tid] = (x, y)
        return players

    @staticmethod
    def _normalise(raw: Dict, timestamp: float, frame: int) -> Dict:
        """Map FootballEventDetector output → canonical schema."""
        team_raw = raw.get("team", raw.get("fouling_team", ""))
        team_map = {"home": "team_a", "away": "team_b"}
        team = team_map.get(str(team_raw).lower(), team_raw)

        etype = raw.get("type", "unknown")

        # Primary actor
        player_id = (
            raw.get("player")
            or raw.get("from_player")
            or raw.get("fouling_player")
            or ""
        )
        # Secondary actor
        secondary = raw.get("to_player") or raw.get("fouled_player") or ""

        meta: Dict[str, Any] = {}
        for k in ("distance_to_goal", "ball_speed_ms", "position",
                  "distance", "corner", "severity", "penalty"):
            if k in raw:
                meta[k] = raw[k]

        return {
            "type":                etype,
            "timestamp":           timestamp,
            "frame":               frame,
            "team":                team,
            "player_id":           player_id,
            "secondary_player_id": secondary,
            "confidence":          0.8,      # rule-based is deterministic
            "metadata":            meta,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ML-based detector stub
# ═══════════════════════════════════════════════════════════════════════════════

class MLEventDetector(BaseEventDetector):
    """
    Stub for a trained action-recognition model (C3D, SlowFast, VideoSwin …).

    To implement:
    1. Load the model in __init__ (model_path, device, confidence_threshold).
    2. In detect(), extract a frame-clip tensor from the snapshot buffer
       (the buffer already represents a sliding window of the right length).
    3. Run inference and map class logits to canonical event dicts.
    4. Register with EventDetectorFactory:
           EventDetectorFactory.register('slowfast', MySlowFastDetector)
    """

    def __init__(self, model_path: str, **kwargs):
        raise NotImplementedError(
            "MLEventDetector is a stub.  "
            "Subclass it and implement detect(), then register with "
            "EventDetectorFactory.register('your_name', YourClass)."
        )

    def detect(self, frame_snapshots):
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble detector
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleEventDetector(BaseEventDetector):
    """
    Combines results from multiple detectors and removes near-duplicate events.

    Deduplication key: (type, round(timestamp / window)) so events of the same
    type within *dedup_window_s* seconds are collapsed to the highest-confidence
    one.
    """

    def __init__(
        self,
        detectors: List[BaseEventDetector],
        dedup_window_s: float = 0.5,
    ):
        self._detectors      = detectors
        self._dedup_window_s = dedup_window_s

    def set_fps(self, fps: float) -> None:
        for d in self._detectors:
            d.set_fps(fps)

    def detect(self, frame_snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_events: List[Dict] = []
        for detector in self._detectors:
            try:
                all_events.extend(detector.detect(frame_snapshots))
            except Exception as exc:
                log.warning("Detector %s failed: %s", type(detector).__name__, exc)
        return self._dedup(all_events)

    def _dedup(self, events: List[Dict]) -> List[Dict]:
        seen: Dict[tuple, Dict] = {}
        for e in events:
            bucket = round(e.get("timestamp", 0) / self._dedup_window_s)
            key    = (e.get("type", ""), bucket, e.get("team", ""))
            if key not in seen or e.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = e
        return list(seen.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

class EventDetectorFactory:
    """
    Create event detectors from a configuration dict.

    Built-in types
    --------------
    'rule_based' — RuleBasedEventDetector (default, no extra deps)
    'ensemble'   — EnsembleEventDetector  (config must include 'detectors' list)
    'ml'         — MLEventDetector        (stub, must be subclassed first)

    Custom detectors
    ----------------
    EventDetectorFactory.register('my_detector', MyDetectorClass)
    cfg = {'type': 'my_detector', 'model_path': '/path/to/weights', ...}
    detector = EventDetectorFactory.create(cfg)

    All extra keys in *cfg* (beyond 'type') are forwarded as kwargs to the
    detector's __init__.
    """

    _registry: Dict[str, type] = {
        "rule_based": RuleBasedEventDetector,
        "ensemble":   EnsembleEventDetector,
        "ml":         MLEventDetector,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseEventDetector:
        cfg  = dict(config)
        name = cfg.pop("type", "rule_based").lower()

        if name not in cls._registry:
            raise ValueError(
                f"Unknown event detector type '{name}'. "
                f"Available: {sorted(cls._registry)}. "
                f"Use EventDetectorFactory.register() to add custom types."
            )

        if name == "ensemble":
            sub_cfgs   = cfg.pop("detectors", [{"type": "rule_based"}])
            sub        = [cls.create(c) for c in sub_cfgs]
            return EnsembleEventDetector(sub, **cfg)

        return cls._registry[name](**cfg)

    @classmethod
    def register(cls, name: str, detector_class: type) -> None:
        """Register a custom detector class so it can be created by name."""
        if not issubclass(detector_class, BaseEventDetector):
            raise TypeError(
                f"{detector_class.__name__} must subclass BaseEventDetector"
            )
        cls._registry[name.lower()] = detector_class
        log.info("EventDetectorFactory: registered '%s'", name)
