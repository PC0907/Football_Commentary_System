# Commentary — Template Generator & TTS Audio Pipeline

This directory contains the **standalone commentary generation script** that reads a JSON
events file and produces:
- An **SRT subtitle file** (timed text captions).
- **TTS audio commentary** (using `pyttsx3` / system speech engine).
- Optionally, an audio-mixed video (requires `ffmpeg` and `pydub`).

The production commentary generator lives in `pipeline/commentary/generator.py`.

---

## Purpose

Once events are detected (pass, shot, goal, corner, foul, free kick), the commentary system
must produce human-readable descriptions for broadcast. This directory explores:

1. **Template-based generation**: hard-coded phrase templates with slots for team name,
   player name, distance, etc.
2. **TTS audio output**: converting text commentary to spoken audio.
3. **SRT export**: exporting timestamped subtitles for video players.

---

## Files

| File | Description |
|------|-------------|
| `audio_generator.py` | Full standalone commentary pipeline. Reads `events.json`, generates commentary text via templates, synthesises TTS audio (pyttsx3), mixes audio onto video (pydub + ffmpeg). |

---

## Input Format

`audio_generator.py` reads a JSON file with a `predictions` array matching the SoccerNet
event format:

```json
{
  "predictions": [
    {
      "gameTime": "1 - 12:34",
      "label": "PASS",
      "team": "home",
      "confidence": 0.91
    },
    ...
  ]
}
```

The `gameTime` field is parsed to derive the subtitle start timestamp. The `label` field
selects the commentary template. Player names (if available) are looked up from the optional
CSV file (`--player-csv`).

---

## Commentary Templates

Each event type has 3–5 template strings with `{player}`, `{team}`, `{distance_to_goal}`,
and `{position}` slots filled at runtime. Templates are chosen randomly per event to
avoid repetition.

Example templates:

| Event | Template |
|-------|---------|
| PASS | `"{player} plays a sharp pass forward for {team}."` |
| SHOT | `"Here comes a shot from {player}! {distance_to_goal:.0f} metres out."` |
| GOAL | `"GOAL! {player} puts it in the back of the net! {team} are celebrating!"` |
| CORNER | `"Corner kick for {team}. {player} steps up to take it."` |
| FREE KICK | `"{team} with a free kick in a dangerous position."` |
| FOUL | `"Foul by {player} of {team} — referee stops play."` |

---

## How It Was Built

### Step 1 — SRT generation
The simplest output is a `.srt` file. Each event gets one subtitle entry:
```
1
00:12:34,000 --> 00:12:38,000
Goal! The away team are celebrating!
```
`audio_generator.py` computes duration as `min(4, next_event_time - this_event_time)` seconds.

### Step 2 — TTS with pyttsx3
`pyttsx3` drives the OS TTS engine (eSpeak on Linux, AVFoundation on macOS, SAPI on Windows).
Each commentary line is synthesised to a temporary `.wav`, then combined into a single audio
track.

### Step 3 — Video mixing with ffmpeg
`subprocess.run(["ffmpeg", "-i", video, "-i", audio, "-c:v", "copy", "-c:a", "aac", out])`
mixes the TTS audio track onto the video. `pydub` is used to concatenate multiple per-event
WAV files with silence padding.

### Ported to production
The template system from `audio_generator.py` was ported to `pipeline/commentary/generator.py`
as `TemplateCommentaryGenerator`. The TTS and ffmpeg mix steps are not yet in the real-time
pipeline (they operate offline on a completed video).

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| pyttsx3 blocks the main thread | Synchronous TTS synthesis | Run TTS in a subprocess; or use async TTS with a queue |
| Audio length doesn't match event spacing | pyttsx3 speech rate varies by OS voice | Set fixed rate: `engine.setProperty('rate', 150)` |
| SRT timestamps off by ±500 ms | `gameTime` parse drops sub-second precision | Round to nearest second; accept that SRT subtitles are approximate |
| ffmpeg not found on some systems | System dependency not installed | Added `shutil.which("ffmpeg")` check with a helpful error message |
| Same template chosen repeatedly | `random.choice` without shuffle | Moved to `random.sample` cycling through all templates before repeating |

---

## What Still Needs Fixing / Future Work

- [ ] **Wire into the real-time pipeline**: `app/processor_thread.py` emits events via
  `commentary_signal`. Connect these to `TemplateCommentaryGenerator` and display text in
  the UI's commentary panel (already wired) but also write to an SRT buffer for export.
- [ ] **LLM-based commentary** (`PhiCommentaryGenerator` in `pipeline/commentary/generator.py`):
  use a small on-device language model (Phi-3-mini, Gemma-2B) to generate more natural,
  varied commentary. The factory and stub are already in place.
- [ ] **TTS export button in the UI**: let the user export the final video with synthesised
  audio commentary after processing is complete.
- [ ] **Multi-language templates**: add commentary templates for Arabic, French, Spanish to
  support international broadcast audiences.
- [ ] **Commentary pacing**: avoid back-to-back commentary events within 3 seconds (e.g.,
  if a shot is immediately followed by a corner). Add a cooldown between commentary outputs.
