# UAV Sliding Window Vegetation Segmentation

Sliding window inference with vote-based fusion for high-resolution UAV vegetation mapping.

Paper: Smart Integration of Sliding Window and Vote-Based Fusion: Advancing UAV-Based Instance Segmentation with YOLOv8  
DOI:https://doi.org/10.1016/j.rsase.2026.101994
Author:Ilias Jennaoui — G2E Lab, Sultan Moulay Slimane University, Morocco

## Problem
Direct inference on high-resolution UAV images (3000×4000+ px) causes severe scale mismatch, plants appear 4.7× smaller than during training. This pipeline solves it using a sliding window approach that maintains vegetation at training scale.

## Method
- Window: 283×283 px — Stride: 85 px — 70% overlap
- Vote-based fusion: argmax of per-class vote counts across overlapping patches
- YOLOv8 instance segmentation backbone

## Species
Lentisque · Chêne-vert · Thuya · Oxycèdre (Middle Atlas, Morocco)

## Install
pip install ultralytics opencv-python matplotlib numpy
## Usage
Edit `IMAGE_MASK_PAIRS` and `MODEL_PATH` in `sliding_window_inference.py` then:

python sliding_window_inference.py

## Results
Published in Remote Sensing Applications: Society and Environment (RSASE), 2026.
