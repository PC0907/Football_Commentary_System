"""
Commentary generation — multiple backends behind one async interface.

Architecture
------------
BaseCommentaryGenerator  (ABC)
├── TemplateCommentaryGenerator   Fast synchronous fallback — template strings +
│                                 random selection; no model required.
└── PhiCommentaryGenerator        Fine-tuned Phi-3.5-Mini (3.8B) via QLoRA.
                                  Runs in a daemon thread; caller submits events
                                  and polls for results — never blocks the
                                  VideoProcessor pipeline thread.

CommentaryGeneratorFactory.create(config: dict) → BaseCommentaryGenerator

Phi-3.5-Mini specifics
-----------------------
Model  : microsoft/Phi-3.5-mini-instruct  (or a local fine-tuned checkpoint)
Quant  : 4-bit NF4 QLoRA via bitsandbytes (GPU required for quantisation;
         CPU fallback uses full float32 on smaller context)
LoRA   : optional adapter loaded with peft.PeftModel — pass lora_path=
Prompt : few-shot + chain-of-thought in Phi's <|user|>/<|assistant|> format
         Model is asked to emit THINKING: then COMMENTARY: — only the latter
         is returned to the UI.

Usage
-----
gen = CommentaryGeneratorFactory.create({'type': 'phi'})
gen.submit({'type': 'shot', 'team': 'team_a', 'player_id': 9,
            'metadata': {'ball_speed_ms': 22.3, 'distance_to_goal': 14.5}})
line = gen.get_nowait()  # → "Number 9 unleashes a thunderbolt…" or None
gen.shutdown()
"""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Team-label helpers ─────────────────────────────────────────────────────────
_TEAM_DISPLAY = {"team_a": "the home side", "team_b": "the away side"}


def _team(event: Dict) -> str:
    return _TEAM_DISPLAY.get(event.get("team", ""), "a team")


def _pid(event: Dict) -> str:
    pid = event.get("player_id", "")
    return f"number {pid}" if pid else "a player"


def _spid(event: Dict) -> str:
    pid = event.get("secondary_player_id", "")
    return f"number {pid}" if pid else "a teammate"


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class BaseCommentaryGenerator(ABC):
    """
    Common async interface for all commentary generators.

    submit(event)    — add an event to the generation queue (non-blocking).
    get_nowait()     — return the next ready commentary string, or None.
    shutdown()       — stop background threads gracefully.
    """

    @abstractmethod
    def submit(self, event: Dict[str, Any]) -> None: ...

    @abstractmethod
    def get_nowait(self) -> Optional[str]: ...

    def shutdown(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Template-based (fast synchronous fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class TemplateCommentaryGenerator(BaseCommentaryGenerator):
    """
    Instant commentary using hand-crafted template strings.
    Each event type has several templates; one is chosen at random and
    formatted with event data.  Results go straight into the output queue.
    """

    _TEMPLATES: Dict[str, List[str]] = {
        "shot": [
            "{pid} pulls the trigger from {dist:.0f} metres — the keeper scrambles!",
            "A ferocious shot from {pid}, {speed:.0f} km/h — straight at goal!",
            "{pid} unleashes a strike from distance — will it find the net?",
            "Effort on goal from {pid}! The goalkeeper will be tested here.",
            "Strike from {pid}! A powerful attempt at goal for {team}.",
        ],
        "pass": [
            "{pid} plays it to {spid} — tidy work in midfield.",
            "A composed pass from {pid}, releasing {spid} into space.",
            "Good vision from {pid} — {spid} is played in.",
            "{pid} finds {spid} with a neat ball. {team} keeping it moving.",
            "The ball is worked to {spid} by {pid}. Possession maintained.",
        ],
        "goal": [
            "GOAL! {pid} finds the back of the net! {team} take the lead!",
            "It's in! {pid} scores a wonderful goal for {team}!",
            "GOOOAL! What a moment! {pid} puts {team} ahead!",
            "The ball is in the net! {pid} has done it for {team}!",
        ],
        "corner_kick": [
            "Corner for {team}. {pid} will deliver this one.",
            "{team} earn a corner — a chance to put the ball in the box.",
            "The referee signals a corner kick. {team} fancy their chances.",
            "Corner kick awarded to {team}. Danger in the box.",
        ],
        "free_kick": [
            "Free kick to {team} in a promising position.",
            "{team} have a free kick — {pid} is standing over it.",
            "Play restarts with a free kick for {team}.",
            "The ball is placed for the free kick. {team} looking to make it count.",
        ],
        "foul": [
            "Foul! The referee stops play. {team} give away possession.",
            "That's a foul — the official has no hesitation in blowing up.",
            "Foul committed by {pid}. The referee shows his authority.",
            "{pid} brings down an opponent. The whistle goes.",
        ],
        "penalty": [
            "PENALTY! The referee points to the spot! Drama for {team}!",
            "It's a penalty! {pid} has been brought down in the box!",
            "Spot kick awarded! The tension rises here at the ground.",
            "Penalty given! {team} have a huge chance to score.",
        ],
        "free_kick_restart": [
            "{team} restart with the free kick — {pid} moves it on quickly.",
            "Quick free kick from {pid} catches the defence off guard.",
        ],
    }

    _FILLER: List[str] = [
        "The ball continues to be worked around the midfield.",
        "Both sides are probing for an opening.",
        "Tight play in the middle of the park.",
        "Possession exchanged as both teams look to build.",
        "The tactical battle continues in the centre of the field.",
    ]

    def __init__(self) -> None:
        self._out: queue.SimpleQueue = queue.SimpleQueue()
        self._last_type: str = ""

    def submit(self, event: Dict[str, Any]) -> None:
        line = self._render(event)
        if line:
            self._out.put(line)

    def get_nowait(self) -> Optional[str]:
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def _render(self, event: Dict[str, Any]) -> str:
        etype     = event.get("type", "unknown")
        templates = self._TEMPLATES.get(etype, [])
        if not templates:
            return ""

        meta  = event.get("metadata", {})
        dist  = meta.get("distance_to_goal", 0) or 0
        speed = (meta.get("ball_speed_ms", 0) or 0) * 3.6   # → km/h

        template = random.choice(templates)
        try:
            return template.format(
                pid=_pid(event), spid=_spid(event),
                team=_team(event), dist=dist, speed=speed,
            )
        except (KeyError, ValueError):
            return template.split("{")[0].strip() + "."


# ═══════════════════════════════════════════════════════════════════════════════
# Phi-3.5-Mini via QLoRA (async daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════

class PhiCommentaryGenerator(BaseCommentaryGenerator):
    """
    Football commentary using a fine-tuned Phi-3.5-Mini (3.8B) model.

    The model is loaded once in __init__ (in the calling thread).
    Inference runs in a background daemon thread so the VideoProcessor
    pipeline is never stalled.

    Parameters
    ----------
    model_id      : HuggingFace model id or local path to base model.
                    Default: "microsoft/Phi-3.5-mini-instruct"
    lora_path     : Local path to a QLoRA adapter (PEFT format).
                    If None, the base model is used with few-shot prompting.
    device        : 'cuda', 'cpu', or 'auto' (default 'auto').
    load_in_4bit  : Enable 4-bit NF4 quantisation (requires GPU + bitsandbytes).
    max_new_tokens: Maximum tokens to generate per commentary line (default 120).
    temperature   : Sampling temperature (default 0.75).
    queue_maxsize : Maximum number of pending events before oldest is dropped.
    """

    MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

    # ── System prompt ──────────────────────────────────────────────────────────
    _SYSTEM = (
        "You are a professional football commentator for live television. "
        "Generate concise, vivid, technically accurate commentary (1-2 sentences). "
        "Use proper football terminology. Vary your language — never repeat a phrase "
        "you have already used in this match. Be excited for goals and shots, measured "
        "for passes and free kicks."
    )

    # ── Few-shot examples (event JSON → THINKING → COMMENTARY) ────────────────
    _FEW_SHOT: List[Dict] = [
        {
            "event": {
                "type": "shot", "team": "team_a", "player_id": "9",
                "metadata": {"ball_speed_ms": 22.3, "distance_to_goal": 14.5},
            },
            "thinking": (
                "Powerful shot (22 m/s ≈ 80 km/h) from 14.5 m — close range, "
                "dangerous. Build excitement; mention the speed and audacity."
            ),
            "commentary": (
                "Number 9 lets fly from just inside the box! "
                "A lightning strike at 80 kilometres per hour — the keeper must react!"
            ),
        },
        {
            "event": {
                "type": "pass", "team": "team_b", "player_id": "8",
                "secondary_player_id": "11",
                "metadata": {"distance": 28.4},
            },
            "thinking": (
                "Long pass (28 m) — likely a switch of play or through-ball. "
                "Describe the vision and weight of the ball."
            ),
            "commentary": (
                "A precise 28-metre ball from number 8 splits the midfield, "
                "finding number 11 in acres of space on the flank."
            ),
        },
        {
            "event": {
                "type": "goal", "team": "team_a", "player_id": "9",
                "metadata": {"distance_to_goal": 11.2},
            },
            "thinking": (
                "GOAL! Maximum excitement. Mention the player, team, "
                "and the crowd reaction. Short, punchy sentences."
            ),
            "commentary": (
                "GOOOAL! Number 9 finds the back of the net! "
                "The home side take the lead and the stadium erupts!"
            ),
        },
        {
            "event": {
                "type": "corner_kick", "team": "team_b", "player_id": "7",
                "metadata": {},
            },
            "thinking": (
                "Corner kick — set-piece opportunity. "
                "Build anticipation of the delivery into the box."
            ),
            "commentary": (
                "Corner for the away side. Number 7 steps up to swing this one "
                "into a crowded penalty area."
            ),
        },
        {
            "event": {
                "type": "foul", "team": "team_a", "player_id": "4",
                "secondary_player_id": "10",
                "metadata": {"severity": 1.8},
            },
            "thinking": (
                "Foul — play is stopped. Describe the challenge and the referee's "
                "decision without being inflammatory."
            ),
            "commentary": (
                "The referee's whistle halts play — number 4 catches number 10 "
                "late, and a free kick is rightly awarded."
            ),
        },
        {
            "event": {
                "type": "penalty", "team": "team_b", "player_id": "11",
                "metadata": {},
            },
            "thinking": (
                "Penalty — highest drama. Build maximum tension. "
                "Short sentences, exclamation."
            ),
            "commentary": (
                "Penalty! The referee points to the spot! "
                "Number 11 was brought down inside the area — "
                "enormous pressure on the goalkeeper now."
            ),
        },
    ]

    def __init__(
        self,
        model_id: str = MODEL_ID,
        lora_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 120,
        temperature: float = 0.75,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        queue_maxsize: int = 30,
    ) -> None:
        self._max_new_tokens    = max_new_tokens
        self._temperature       = temperature
        self._top_p             = top_p
        self._rep_penalty       = repetition_penalty

        self._pending: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._ready:   queue.Queue = queue.Queue(maxsize=200)
        self._stop_flag            = threading.Event()

        # Build the prompt prefix once (few-shot block is static)
        self._few_shot_block = self._build_few_shot_block()

        # Load model (may raise — caller should catch and fall back)
        self._model, self._tokenizer = self._load_model(
            model_id, lora_path, device, load_in_4bit
        )

        # Start background inference thread
        self._worker = threading.Thread(
            target=self._inference_loop, daemon=True, name="phi-commentary"
        )
        self._worker.start()
        log.info("PhiCommentaryGenerator: model loaded, inference thread started")

    # ── Public interface ───────────────────────────────────────────────────────

    def submit(self, event: Dict[str, Any]) -> None:
        """Queue an event for commentary generation (drops if queue is full)."""
        try:
            self._pending.put_nowait(event)
        except queue.Full:
            # Drop the oldest event and add the new one
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pending.put_nowait(event)
            except queue.Full:
                pass

    def get_nowait(self) -> Optional[str]:
        """Return the next ready commentary line, or None."""
        try:
            return self._ready.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        """Signal the inference thread to stop and wait up to 3 s."""
        self._stop_flag.set()
        self._pending.put(None)   # unblock the blocking get()
        self._worker.join(timeout=3.0)

    # ── Model loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(model_id, lora_path, device, load_in_4bit):
        """
        Load Phi-3.5-mini-instruct with optional 4-bit QLoRA quantisation.

        4-bit path   : requires `bitsandbytes` + CUDA.
        CPU fallback : loads in float32 — slow but functional for testing.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        use_4bit = load_in_4bit and torch.cuda.is_available()

        if use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map="auto" if device == "auto" else device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
                if PhiCommentaryGenerator._has_flash_attn()
                else "eager",
            )
        else:
            log.warning(
                "PhiCommentaryGenerator: CUDA not available or load_in_4bit=False. "
                "Loading in float32 — inference will be slow."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        # Load LoRA adapter (fine-tuned sports commentary weights)
        if lora_path:
            from peft import PeftModel
            log.info("PhiCommentaryGenerator: loading LoRA adapter from %s", lora_path)
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()   # merge for faster inference

        model.eval()
        return model, tokenizer

    @staticmethod
    def _has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_few_shot_block(self) -> str:
        """Build the static few-shot examples block (called once at init)."""
        lines = ["## Examples\n"]
        for ex in self._FEW_SHOT:
            ev_json = json.dumps(ex["event"], separators=(", ", ": "))
            lines.append(f"EVENT: {ev_json}")
            lines.append(f"THINKING: {ex['thinking']}")
            lines.append(f"COMMENTARY: {ex['commentary']}\n")
        return "\n".join(lines)

    def _build_prompt(self, event: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Construct the ChatML messages list for Phi-3.5-mini-instruct.

        Chain-of-thought: the model is asked to emit THINKING: (internal
        analysis) before COMMENTARY: (the actual output line).  Only the
        COMMENTARY: line is extracted from the response.
        """
        ev_json = json.dumps(
            {
                "type":       event.get("type", "unknown"),
                "team":       event.get("team", ""),
                "player_id":  event.get("player_id", ""),
                "secondary_player_id": event.get("secondary_player_id", ""),
                **event.get("metadata", {}),
            },
            separators=(", ", ": "),
        )

        user_content = (
            f"{self._few_shot_block}\n"
            "## Current Event\n\n"
            f"EVENT: {ev_json}\n\n"
            "Analyze this event step by step, then write the commentary.\n"
            "Use this exact format:\n"
            "THINKING: <your analysis>\n"
            "COMMENTARY: <one or two sentences of live commentary>"
        )

        return [
            {"role": "system",    "content": self._SYSTEM},
            {"role": "user",      "content": user_content},
        ]

    # ── Inference ──────────────────────────────────────────────────────────────

    def _generate(self, event: Dict[str, Any]) -> Optional[str]:
        """Run one forward pass and extract the COMMENTARY: line."""
        import torch

        messages = self._build_prompt(event)

        # Phi-3.5-mini-instruct has a built-in chat template
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._rep_penalty,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids   = output_ids[0, input_ids.shape[1]:]
        raw_text  = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        return self._extract_commentary(raw_text)

    @staticmethod
    def _extract_commentary(raw: str) -> Optional[str]:
        """
        Extract the COMMENTARY: line from the model's raw output.

        Handles:
        - 'COMMENTARY: text'
        - 'Commentary: text' (case-insensitive)
        - Missing COMMENTARY: tag → use last non-empty line as fallback
        """
        for line in raw.splitlines():
            if line.lower().startswith("commentary:"):
                text = line.split(":", 1)[1].strip()
                if text:
                    return text
        # Fallback: return last non-empty line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else None

    # ── Background thread ──────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """Daemon thread: read events → generate commentary → emit to output queue."""
        while not self._stop_flag.is_set():
            try:
                event = self._pending.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:   # shutdown sentinel
                break

            try:
                commentary = self._generate(event)
                if commentary:
                    self._ready.put(commentary)
            except Exception as exc:
                log.warning(
                    "PhiCommentaryGenerator: generation failed for event %s — %s",
                    event.get("type"), exc,
                    exc_info=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

class CommentaryGeneratorFactory:
    """
    Create commentary generators from a configuration dict.

    Built-in types
    --------------
    'phi'      — PhiCommentaryGenerator (Phi-3.5-Mini via QLoRA).
                 Falls back to 'template' if transformers / GPU not available.
    'template' — TemplateCommentaryGenerator (no model, always available).

    Custom generators
    -----------------
    CommentaryGeneratorFactory.register('my_llm', MyLLMGenerator)

    Config keys for 'phi'
    ---------------------
    model_id       : str   (default: microsoft/Phi-3.5-mini-instruct)
    lora_path      : str   (default: None — uses base model)
    load_in_4bit   : bool  (default: True)
    max_new_tokens : int   (default: 120)
    temperature    : float (default: 0.75)
    """

    _registry: Dict[str, type] = {
        "phi":      PhiCommentaryGenerator,
        "template": TemplateCommentaryGenerator,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseCommentaryGenerator:
        cfg  = dict(config)
        name = cfg.pop("type", "phi").lower()

        if name not in cls._registry:
            raise ValueError(
                f"Unknown commentary generator type '{name}'. "
                f"Available: {sorted(cls._registry)}."
            )

        klass = cls._registry[name]

        if klass is PhiCommentaryGenerator:
            return cls._try_phi(cfg)

        return klass(**cfg)

    @classmethod
    def _try_phi(cls, cfg: Dict[str, Any]) -> BaseCommentaryGenerator:
        """
        Try to load PhiCommentaryGenerator; fall back to TemplateCommentaryGenerator
        if transformers, bitsandbytes, or CUDA are unavailable.
        """
        try:
            import transformers  # noqa: F401
            gen = PhiCommentaryGenerator(**cfg)
            return gen
        except ImportError as exc:
            log.warning(
                "PhiCommentaryGenerator not available (%s). "
                "Falling back to TemplateCommentaryGenerator.", exc
            )
        except Exception as exc:
            log.warning(
                "PhiCommentaryGenerator failed to load (%s). "
                "Falling back to TemplateCommentaryGenerator.", exc
            )
        return TemplateCommentaryGenerator()

    @classmethod
    def register(cls, name: str, generator_class: type) -> None:
        """Register a custom generator class so it can be created by name."""
        if not issubclass(generator_class, BaseCommentaryGenerator):
            raise TypeError(
                f"{generator_class.__name__} must subclass BaseCommentaryGenerator"
            )
        cls._registry[name.lower()] = generator_class
        log.info("CommentaryGeneratorFactory: registered '%s'", name)
