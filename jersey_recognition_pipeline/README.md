# Jersey Recognition Pipeline

This repository contains a modular pipeline for jersey recognition as part of a commentary system. The pipeline consists of multiple stages, including object detection, player crop extraction, legibility classification, and torso region extraction using VitPose. 

## Table of Contents
- [Pipeline Structure](#pipeline-structure)
- [Dependencies and Installations](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Components](#components)
- [Configuration](#configuration)
- [Results](#results)

## Overview
The pipeline is designed to automate the process of recognizing jersey numbers in football videos by following these steps:
1. **Object Detection**: Detect players in the video frames.
2. **Player Crop Extraction**: Extract the detected player regions from each frame.
3. **Legibility Classification**: Filter out player crops where the jersey number is not clearly visible.
4. **Torso Extraction**: Use VitPose to extract the torso region for better jersey recognition.

### Installation and Setup
```sh
pip install -r requirements.txt
python setup.py
```
Example Run Script:
```sh
python main.py --input test_video.mp4 --output my_results/
```
Expected Output Structure:
```
my_results/
├── crops/
├── legible_crops/
└── torso_crops/
```

## Pipeline Structure
```
Jersey-Recognition-Pipeline/
│
├── README.md                   # Project overview and instructions
├── LICENSE                     # License information              
│
├── src/
│   ├── pipeline/               # Moved from root
│   │   ├── __init__.py
│   │   ├── football_pipeline.py
│   │   ├── object_detector.py
│   │   ├── crop_processor.py
│   │   ├── classifier.py
│   │   └── pose_estimator.py
│   └── video_processor.py      # New location
├── configs/
│   └── paths.py                # Updated content below
│   
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # Pose visualization
│   └── file_utils.py
|
|
├── outputs/
│   ├── keypoints/              # Keypoint visualizations
│   ├── torso_crops/            # Extracted torso crops
│   └── processed_torso/        # Enhanced torso crops
│
├── models/                     # Pre-trained models
│   ├── detection/              # Object detection models
│   ├── legibility/             # Legibility classifier models
│   └── vitpose/                # VitPose model files
|
├── results/                    # Outputs generated from the pipeline
│   ├── detected_videos/        # Videos with object detection results
│   ├── extracted_crops/        # Player crops extracted from videos
│   ├── filtered_crops/         # Crops after legibility filtering
│   └── torso_regions/          # Extracted torso regions
│
└── requirements.txt
└── setup.py
└── verify_installation.py
└── main.py

```
