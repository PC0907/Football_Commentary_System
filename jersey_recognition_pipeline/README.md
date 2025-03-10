# Jersey Recognition Pipeline

This repository contains a modular pipeline for jersey recognition as part of a football analysis system. The pipeline consists of multiple stages, including object detection, player crop extraction, legibility classification, and torso region extraction using VitPose. 

## Table of Contents
- [Overview](#overview)
- [Pipeline Structure](#pipeline-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Components](#components)
- [Configuration](#configuration)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Overview
The pipeline is designed to automate the process of recognizing jersey numbers in football videos by following these steps:
1. **Object Detection**: Detect players in the video frames.
2. **Player Crop Extraction**: Extract the detected player regions from each frame.
3. **Legibility Classification**: Filter out player crops where the jersey number is not clearly visible.
4. **Torso Extraction**: Use VitPose to extract the torso region for better jersey recognition.

## Pipeline Structure
