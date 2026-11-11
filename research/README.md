# research/ — perception and commentary research for the Football Commentary System

This directory holds the research code behind the system's next version. Each component is developed and
benchmarked on public SoccerNet data before it is moved into the production pipeline (`../pipeline`, `../app`).

- **Player and referee tracking:** a Kalman-filter tracker with camera-motion compensation, pitch-plane physics,
  a learned appearance likelihood ratio, physics-feasible tracklet stitching and score-aware association.
- **Camera calibration:** *PitchGeo*, a network that predicts pitch markings as line equations and ellipses,
  each with an uncertainty. A MAP camera solver turns them into a full camera.
- **Ball tracking:** a causal multi-"eye" tracker that uses only current and past frames. The eyes are a full-frame
  detector, a camera-motion-compensated motion detector and a learned temporal fovea network.
- **Commentary:** a small local LLM (Qwen3.5-2B / 4B) fine-tuned with QLoRA to write commentary from a
  textual game state.

The research tracker and calibration are not yet integrated into `../pipeline`.

## Results so far

The numbers come from the scripts in `scripts/`. `notebooks/comparison.ipynb` shows them as charts and
example frames, read from the result files.

### Player and referee tracking (SoccerNet-Tracking, test, 49 sequences)

| Tracker | GT boxes: HOTA | GT boxes: ID switches | YOLO26-m boxes: HOTA | YOLO26-m boxes: ID switches |
|---|---|---|---|---|
| Thesis tracker (before the fix) | 72.1 | 7037 | — | — |
| ByteTrack | 83.6 | 3827 | 60.5 | 3096 |
| OC-SORT + camera-motion compensation | 85.6 | 3710 | 63.1 | 3305 |
| BoT-SORT (no ReID) | 86.3 | 1783 | 61.2 | **1624** |
| **Ours (T4)** | **90.0** | **1232** | **65.4** | 1981 |

**Protocol:**
- **GT boxes** means ground-truth boxes with identities removed, which measures association only.
- **YOLO26-m boxes** come from a detector fine-tuned on SoccerNet-Tracking train.
- In the YOLO26-m setting, every tracker's confidence thresholds were tuned on the same 15 held-out training
  sequences (the detector's validation sequences). Each tracker then ran once on test.
- All trackers get the same boxes, camera-motion compensation and track memory. ReID is off for all of them.

**Significance** (paired over the 49 sequences, with a bootstrap CI and Wilcoxon test):
- GT boxes, ours vs BoT-SORT: +3.8 HOTA [2.8, 5.0], p = 5×10⁻¹³, better on 46 of 49 sequences.
- YOLO26-m boxes, ours vs OC-SORT + CMC: +2.2 HOTA [1.3, 3.1], p = 3×10⁻⁵, better on 33 of 49.

**Limitation:** BoT-SORT still makes fewer ID switches with real boxes.

### Camera calibration (SoccerNet-Calibration-2023, test)

| Method | Final score (completeness × JaC@5) |
|---|---|
| PitchGeo v1 | 0.009 |
| **PitchGeo v2** (smoothly bounded σ + mirror-invariant probabilistic loss) | **0.512** |
| PnLCalib 512×512 (published) | 0.518 |
| PnLCalib 960×540 (published, best single-view) | 0.609 |

- **Ablation:** on the same code, the probabilistic (Laplace NLL) loss halves the validation line error compared
  with a deterministic Huber loss: 22.7 px vs 42.7 px.
- **Metric caveat:** our score uses a re-implementation of the official metric, which has not yet been
  cross-checked against the official script.

### Commentary (SoccerNet-Caption, MatchTime test-align set, 3,267 captions, MatchTime scoring protocol)

| Model | BLEU-4 | CIDEr |
|---|---|---|
| Qwen3.5-2B zero-shot | 0.7 | 3.3 |
| Qwen3.5-2B, QLoRA | 8.7 | 36.2 |
| Qwen3.5-4B, QLoRA | 8.3 | 37.8 |
| MatchVoice (published) | 10.1 | 42.0 |

**Caveat:** the game-state input is built from ground-truth event labels. A live system must detect the events
first.

### Ball (work in progress; 15 held-out tuning sequences, YOLO26-m)

| Method | HOTA | Recall @ 10 px | Precision @ 10 px |
|---|---|---|---|
| Best detection per frame, tuned threshold | 43.1 | 77.5 | 88.7 |
| Causal tracker + motion eye | **44.0** | **78.6** | 86.5 |

**Findings:**
- **Fast balls are the hard case.** Recall drops from 0.89 for slow balls to 0.42 for balls moving faster than
  25 px/frame. Fast balls appear as pale streaks aligned with the motion.
- **Camera-motion compensation hurts the ball.** The broadcast camera follows the ball, so compensating its
  motion doubles the ball's frame-to-frame step.
- **HOTA saturates for the ball.** A 15 px box has a median IoU of 0.66 even when correctly located, so
  distance-based recall and precision are reported as well.

## Setup

```bash
conda create -n fcs-research python=3.11 && conda activate fcs-research
pip install torch==2.6.* torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes accelerate pycocoevalcap ultralytics "rfdetr[train,loggers]" \
            opencv-python scipy scikit-learn pandas matplotlib huggingface_hub kaggle nbformat nbclient ipykernel
pip install --no-deps boxmot==26.0.0 && pip install filterpy lapcpp timm      # baseline trackers
git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval   # HOTA/IDF1 (git-ignored)
```

### Data

Data lives outside the repository and is downloaded from the SoccerNet Hugging Face datasets.

| Path | Contents |
|---|---|
| `~/soccerNet/hf/tracking/{train,test}` | SN-Tracking-2023. Locally, `train` holds only the 15-sequence tuning subset (every 4th official training sequence). |
| `~/soccerNet/hf/calib2023/{valid,test}.zip` | SN-Calibration-2023 |
| `~/soccerNet/hf/labels/` | SoccerNet Labels-v2 and Labels-caption. Override with `SN_LABELS_ROOT`. |
| `~/soccerNet/matchtime_hf/` | MatchTime SN-Caption-test-align. Override with `MATCHTIME_ROOT`. |

Everything under `cache/`, `outputs/`, `logs/` and `third_party/`, plus model weights, is git-ignored.

## Running experiments

Every experiment is a script with command-line arguments. Long scripts report progress, so any of them can run
as a tracked job:

```bash
python scripts/job.py run NAME -- python scripts/<script>.py <args>   # launch detached
python scripts/job.py watch          # live progress, ETA, CPU/GPU/RAM per job
python scripts/job.py status --all   # all jobs incl. finished
python scripts/job.py tail JOB_ID    # last log lines
```

Main entry points (each script's docstring lists its options):

| Task | Commands |
|---|---|
| Tracking inputs | `precompute_snmot.py --split test` (keypoints + ego-motion); `precompute_dets.py` / `infer_dets_local.py` (detections) |
| Tracking | `run_tracking.py --split test --det oracle\|yolo26m_snmot --variants T1 T4 X-botsort ...`; `tune_tracking.py`; `bootstrap_tracking.py` (significance) |
| Calibration | `train_pitchgeo.py`; `eval_pitchgeo.py --ckpt ... --zip .../test.zip --tag NAME` |
| Ball | `run_ball.py --variants B0 O1 O4 ...`; `tune_ball.py`; `sahi_ball_dets.py`; `extract_ball_crops.py` + `train_fovea.py` |
| Detector | `train_detector_snmot.py --arch yolo26m\|rfdetr-m` (has `--smoke` and `--train-hours`) |
| Commentary | `train_commentary_lora.py`; `commentary_llm.py --split test --n 0 [--adapter DIR]`; `run_commentary_baselines.py` |
| Figures | `build_comparison_notebook.py` → `notebooks/comparison.ipynb` |

The `diag_*.py` scripts are diagnostics; each writes a JSON file to `outputs/diag/`.

### Kaggle

`kaggle/push_kernel.py TASK NAME "ARGS"` builds a self-contained kernel with the code embedded and pushes it.
TASK is one of `pitchgeo`, `detector`, `commentary` or `balldata` (CPU only).

- `scripts/kaggle_watch.py` waits for a kernel and downloads its outputs.
- `scripts/kaggle_queue_push.py` retries until a GPU slot is free.
- `--build-only` writes a one-cell notebook that someone else can run on their own account.

Before pushing a kernel:
1. Run the same arguments locally with `--smoke`.
2. Install the kernel's pip packages locally and check the imports.
3. Check the outputs' meaning, not only the exit code.
4. Keep the estimated runtime under Kaggle's 12-hour limit and pass a hard cap (`--train-hours`).

## Layout

```
fcs/                  library code
  filters.py            Kalman / RTS / IMM primitives, constraint projection
  progress.py           progress files read by scripts/job.py
  geometry/             ego-motion, keypoint/line calibration sensors, calibration filter + smoother, camera model
  calib/                PitchGeo model, dataset, pitch model, MAP camera solver, SN-Calibration metric
  tracking/             SNMOT I/O, TrackEval wrapper, player tracker, ball trackers (online, offline, IMM), motion eye
  ball/                 fovea network (CNN and Kalman-prior-attention transformer) and its online eye
  commentary/           caption dataset (game state from Labels-v2) and MatchTime-protocol metrics
  viz.py                helpers for the comparison notebook
scripts/              experiments, tuning, significance tests, diagnostics, Kaggle helpers
kaggle/               kernel builder (build/ and hand-off notebooks are git-ignored)
baselines/            frozen copy of the original production tracker (baseline T0)
notebooks/            comparison.ipynb (generated)
```
