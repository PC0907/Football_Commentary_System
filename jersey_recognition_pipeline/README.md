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

## Pipeline Structure
'''
football-analysis-pipeline/
│
├── README.md                   # Project overview and instructions
├── LICENSE                     # License information
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files and directories to ignore
│
├── src/                        # Source code
│   ├── main.py                 # Entry point for running the full pipeline
│   ├── config/                 # Configuration files
│   ├── utils/                  # Utility scripts (e.g., helper functions, logging)
│   ├── object_detection/       # Object detection on video
│   ├── player_crops/           # Extracting player crops from detected video
│   ├── legibility_classifier/  # Filtering illegible jersey number crops
│   ├── torso_extraction/       # Extracting torso regions using VitPose
│   └── visualization/          # Visualization utilities (optional)
│
├── data/                       # Data for processing
│   ├── raw/                    # Raw input videos
│   ├── processed/              # Processed outputs (player crops, detected events)
│   └── annotations/            # Annotations for evaluation (optional)
│
├── models/                     # Pre-trained models
│   ├── detection/              # Object detection models
│   ├── legibility/             # Legibility classifier models
│   └── vitpose/                # VitPose model files
│
├── results/                    # Outputs generated from the pipeline
│   ├── detected_videos/        # Videos with object detection results
│   ├── extracted_crops/        # Player crops extracted from videos
│   ├── filtered_crops/         # Crops after legibility filtering
│   └── torso_regions/          # Extracted torso regions
│
└── docs/                       # Documentation
    ├── architecture.md         # Project architecture details
    ├── pipeline.md             # Explanation of the processing pipeline
    └── api_reference.md        # API documentation for modules
'''
