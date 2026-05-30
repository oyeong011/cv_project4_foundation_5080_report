# CV Project #4 Foundation Model Report

This repository contains the final report PDF for CV Project #4: Foundation model based cocktail robot perception experiment.

## Main file

- `CV_Project4_Foundation_Model_Report.pdf`

## What was applied

- GroundingDINO open-vocabulary detection on the test images
- OWLv2 zero-shot detector fallback
- YOLOv8-seg `best.pt` prediction-only baseline where available
- HSV-based dispenser color/order estimation
- Cup center/top/grasp-candidate perception feature extraction
- Manual-GT subset evaluation template and diagnostics

## Important limitation

The current dataset folder contains no YOLO `.txt` GT label files. Therefore supervised precision/recall/mAP and segmentation IoU are intentionally marked `not measured` rather than fabricated. SAM2 predictor loading hit CUDA OOM, so mask rows are documented as box-mask fallback.

No robot motion, ROS2 service call, MoveIt execution, gripper command, or hardware-control code is included.
