# Jersey Annotator — Labelling Tool for Jersey-Number OCR Training

This directory contains a **PyQt5-based image annotation tool** built specifically for
labelling football player jersey numbers in cropped player images.

The labelled dataset produced here is used to train or fine-tune the OCR classifier in
`experiments/jersey_recognition/`.

---

## Purpose

Jersey number recognition requires a dataset of labelled player-crop images where each
image is tagged with the jersey number it shows. Manual labelling is tedious; this tool
makes it fast by:

- Loading crops from a folder in sequence.
- Accepting a keyboard-typed number label.
- Saving labels to a CSV and optionally augmenting each image (10+ variants) on the fly.

---

## Files

| File | Description |
|------|-------------|
| `annotator.py` | Main PyQt5 GUI application. Entry point. |
| `image_loader.py` | Handles directory scanning, sequential loading, and prev/next navigation. |
| `csv_handler.py` | Reads and writes `annotations.csv`; tracks which images are already labelled. |
| `augmentor.py` | Applies augmentation (brightness, rotation, perspective, blur) to an image and saves `_aug1`, `_aug2`, … variants. |
| `inputImages/` | Put unlabelled jersey crops here before running the tool. |
| `outputImages/` | Annotated (and augmented) images are written here alongside `annotations.csv`. |
| `LICENSE` | MIT licence. |

---

## Quick Start

```bash
pip install opencv-python numpy PyQt5
python annotator.py
```

1. Click **Load Folder** → select `inputImages/`.
2. Click **Select Output Folder** → select `outputImages/`.
3. Type the jersey number with the keyboard → press **Enter** to save.
4. Use **← →** to navigate back and forward.
5. Enable **Augmentation Mode** to generate 10 extra augmented variants per image.

---

## How the Tool Was Built

The annotation requirement arose because no public dataset of football jersey-number crops
at broadcast resolution existed that matched the visual conditions of our video (compression
artefacts, motion blur, small player size).

Steps:
1. Run the object detector on match footage to extract player bounding-box crops.
2. Manually label each crop with its jersey number using this tool.
3. Augmentation mode multiplies each labelled image ×10, giving ~10× more training data
   with realistic noise.
4. The resulting `annotations.csv` feeds the jersey recognition training pipeline.

### Session management

`session_data.json` (auto-generated in the output folder) records the index of the last
labelled image. On next launch, a popup asks to resume from that index — no work is lost.

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Labels not saved if output folder not selected | No validation before write | Added guard: tool disables the `Enter` key until output folder is set |
| Augmented images overwrite originals | Filename collision | Augmented images get suffix `_aug1`, `_aug2`, … |
| Session resume picks wrong image | Index stored relative to sorted filename list | Sort filenames alphabetically before indexing |
| PyQt5 vs PyQt6 conflict | The main app uses PyQt6; this tool uses PyQt5 | This tool is standalone — do not mix into the Qt6 app |

---

## What Still Needs Fixing / Future Work

- [ ] **Auto-OCR suggestion**: run a quick Tesseract or EasyOCR pass on each crop and
  pre-fill the number field so the annotator only needs to confirm/correct rather than
  type from scratch.
- [ ] **Bulk reject mode**: many crops show the player from behind (no visible number).
  Add a single-key "skip / no number" shortcut that marks the crop as unlabelled without
  adding it to the training set.
- [ ] **Quality-filter crops**: crops where the player is too small (< 30 px height) or too
  blurry (Laplacian variance < threshold) should be auto-skipped — they add noise to training.
- [ ] **Port to PyQt6**: the main app uses PyQt6; this tool uses PyQt5. They cannot coexist
  in the same Python process. Port `annotator.py` to PyQt6 to remove the dependency conflict.
