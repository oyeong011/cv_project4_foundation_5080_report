# Final Submission Audit

Generated: 2026-05-30T23:16:45
Overall: PASS

This audit checks perception artifacts only. No robot motion, ROS2 service call, MoveIt execution, gripper command, or hardware-control code is required or generated.

| Check | OK | Evidence |
|---|---:|---|
| `exists:outputs/predictions/foundation_boxes.csv` | True | size=94766 rows=282 |
| `exists:outputs/predictions/foundation_masks.csv` | True | size=94766 rows=282 |
| `exists:outputs/predictions/owlv2_boxes.csv` | True | size=72685 rows=203 |
| `exists:outputs/predictions/owlv2_masks.csv` | True | size=72685 rows=203 |
| `exists:outputs/predictions/cup_geometry.csv` | True | size=13892 rows=57 |
| `exists:outputs/predictions/dispenser_color_order.csv` | True | size=77357 rows=368 |
| `exists:outputs/predictions/remove_candidates.csv` | True | size=85107 rows=485 |
| `exists:outputs/manual_eval/manual_gt_bboxes.csv` | True | size=713 rows=20 |
| `exists:outputs/metrics/label_match_diagnosis.csv` | True | size=850 rows=3 |
| `exists:outputs/metrics/label_weight_diagnosis_summary.csv` | True | size=520 rows=7 |
| `exists:outputs/metrics/manual_detection_metrics.csv` | True | size=319 rows=1 |
| `exists:outputs/predictions/yolo_baseline_predictions.csv` | True | size=1877 rows=12 |
| `exists:outputs/metrics/prediction_summary.csv` | True | size=394 rows=3 |
| `exists:outputs/metrics/experiment_status.csv` | True | size=1528 rows=11 |
| `exists:outputs/metrics/summary_metrics.csv` | True | size=2166 rows=8 |
| `exists:report/CV_Project4_Foundation_Model_Report.md` | True | size=38198 |
| `exists:report/CV_Project4_Foundation_Model_Report.pdf` | True | size=26288 |
| `exists:SUBMISSION_CHECKLIST.md` | True | size=1230 |
| `exists:SUBMISSION_MANIFEST.json` | True | size=7031 |
| `foundation_boxes_nonempty` | True | rows=282 |
| `foundation_masks_nonempty` | True | rows=282 |
| `yolo_baseline_prediction_rows_recorded` | True | rows=12 |
| `manual_gt_evaluation_path_exists` | True | rows=1 |
| `mask_pngs_exist` | True | png_count=491 |
| `visualizations_exist` | True | png_count=116 |
| `qwen_aux_attempt_recorded` | True | parsed_rows=0 |
| `no_robot_control_code_detected` | True | findings=0 |

## Notes

- If GT labels or YOLO `best.pt` are absent, supervised AP/IoU metrics are expected to be marked `not measured`.
- If Qwen2.5-VL or SAM2 cannot complete because of model download, VRAM, or package constraints, the report keeps the attempt and fallback status explicit.
