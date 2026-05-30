# Submission Checklist

- Project root: `/home/ssu/cv_project4_foundation_5080`
- Report: `report/CV_Project4_Foundation_Model_Report.md`
- Notebook: `notebooks/CV_Project4_Foundation_Cocktail_Robot_5080.ipynb`
- GroundingDINO prediction rows: 282
- OWLv2 fallback prediction rows: 203
- Cup geometry rows: 57
- Dispenser color/order rows: 368
- Remove candidate rows: 485
- YOLO baseline prediction rows: 12
- Manual GT metric rows: 1
- Experiment status table: `outputs/metrics/experiment_status.csv`
- Final audit: `FINAL_SUBMISSION_AUDIT.md` and `outputs/metrics/final_audit.csv`
- GT labels: not available in current dataset, so AP/IoU metrics are not measured.
- YOLO best.pt: found automatically; prediction-only baseline rows generated, but supervised mAP is not measured because GT labels are absent.
- SAM2: package installed/import checked, but predictor load failed on RTX 5080 due CUDA OOM / CPU no-GPU path; box-only fallback masks are generated and counted separately.
- Qwen2.5-VL: attempted as auxiliary VLM; no parsed rows within timeout.
- Safety: no ROS2 service call, MoveIt execution, gripper command, robot motion command, or hardware control code included.

See `SUBMISSION_MANIFEST.json` for file hashes.
