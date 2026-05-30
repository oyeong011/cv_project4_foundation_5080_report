# CV Project #4: Foundation Model 기반 칵테일 로봇 물체 인식 및 조작 후보점 추정 실험

생성일: 2026-05-30

## 연구 요약

- **적용 내용:** RTX 5080 로컬 환경에서 GroundingDINO open-vocabulary detector, OWLv2 zero-shot fallback detector, YOLOv8-seg fine-tuned weight의 prediction-only baseline, HSV 기반 디스펜서 색상/순서 추정, 컵 중심/상단/조작 후보점 계산을 실제 test 43장에 적용했다.
- **생성 산출물:** GroundingDINO 계열 282개 box/mask row, OWLv2 203개 row, YOLO baseline prediction-only 12개 row, cup geometry 57개 row, dispenser color/order 368개 row, remove_candidate 485개 row를 생성했다.
- **측정 가능 지표:** GT 없이도 측정 가능한 image coverage, prediction count, confidence distribution, latency, downstream feature 생성 여부를 표로 기록했다. 또한 수동 GT 10~20장만 추가하면 precision/recall/F1/AP50/AP50-95를 계산하는 manual subset 평가 도구를 만들었다.
- **GT 부재로 산출하지 않은 지표:** 현재 `labels/train|val|test`에는 YOLO `.txt` GT가 없으므로 supervised precision/recall/mAP50/mAP50-95와 mask IoU는 계산하지 않았다. 이 값을 임의로 만들지 않고 `not measured`로 둔다.
- **SAM2 실행 결과 해석:** SAM2 predictor import는 확인했지만 predictor load에서 CUDA OOM이 발생했다. 따라서 이번 CSV의 mask는 실제 SAM2 segmentation mask가 아니라 box-mask fallback이며, 보고서에서 `sam2_real_masks=0`, `box_fallback_masks=282/203`으로 분리했다.
- **Qwen2.5-VL 실행 결과 해석:** Qwen2.5-VL은 보조 VLM 실험으로 시도했지만 parsed row가 0개라 성공 결과로 해석하지 않고 보조 실험의 한계로 기록했다.
- **실험 범위:** 이 프로젝트는 perception CSV와 시각화만 생성하며, 로봇 제어/하드웨어 제어 코드는 포함하지 않는다.

## 방법론 선택 근거 및 평가 범위

1. **GroundingDINO 적용 근거:** 텍스트 프롬프트로 `cup`, `lid`, `dispenser` 같은 open-vocabulary 객체 후보를 찾는 목적이므로 fine-tuning 없이 탐지하는 foundation detector가 필요했다.
2. **SAM2 적용 근거:** box prompt로 mask를 생성해 bbox 중심보다 나은 컵 geometry를 얻는 것이 원래 목적이었다. 다만 이번 실행에서는 OOM 때문에 fallback을 사용했으므로 SAM2 성능으로 과장하지 않았다.
3. **OWLv2 fallback 적용 근거:** GroundingDINO가 실패하거나 prompt/threshold에 민감할 때 비교 가능한 zero-shot detection fallback으로 사용했다.
4. **Qwen2.5-VL 보조 실험 설정 근거:** VLM JSON 출력은 재현성과 bbox metric 안정성이 detector보다 낮으므로 정량 detector가 아니라 색상/순서/설명 보조 가능성만 확인하는 위치로 두었다.
5. **GT 기반 mAP 미산출 사유:** 평가 코드가 없는 것이 아니라, 현재 데이터셋에 GT `.txt` 파일이 없어서 예측-정답 매칭 자체가 불가능하다. 대신 label/weight 진단 CSV와 manual GT subset 평가 도구를 추가했다.

## 1. 실험 개요

본 실험은 칵테일 제조 로봇의 perception pipeline에서 컵, 뚜껑, 디스펜서를 인식하고 조작 후보점을 계산하기 위해 foundation model 기반 open-vocabulary detection 및 promptable segmentation을 적용한 실험이다.

본 과제의 범위는 perception 성능 측정이므로 실제 로봇 motion command는 생성하지 않았고, CSV 형태의 perception 결과만 산출하였다.

## 2. 실험 목적

YOLOv8-seg fine-tuned supervised baseline과 GroundingDINO + SAM2 zero-shot/promptable pipeline을 비교한다. 다만 현재 데이터셋의 GT label 파일이 없기 때문에 완전한 supervised benchmark가 아니라, foundation model 기반 perception pipeline의 실행 가능성, prediction coverage, latency, confidence 분포, downstream feature 생성 가능성을 중심으로 평가한다.

## 3. RTX 5080 실험 환경

표 1. RTX 5080 환경

| python_version                                     | platform                                     | torch_version   | cuda_available   |   torch_cuda_version |   cudnn_version | gpu_name                |   gpu_total_memory_mb |   gpu_allocated_memory_mb | nvidia_smi                                  |
|:---------------------------------------------------|:---------------------------------------------|:----------------|:-----------------|---------------------:|----------------:|:------------------------|----------------------:|--------------------------:|:--------------------------------------------|
| 3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0] | Linux-6.8.0-40-generic-x86_64-with-glibc2.35 | 2.11.0+cu130    | True             |                   13 |           91900 | NVIDIA GeForce RTX 5080 |               15831.6 |                         0 | NVIDIA GeForce RTX 5080, 580.142, 16303 MiB |

## 4. 데이터셋 구조 및 GT label 진단

실제 데이터셋은 `/home/ssu/Downloads/yolo_cup_dataset` 아래 `images/train|val|test`, `labels/train|val|test`, `cup.yaml` 구조를 우선 지원한다. 라벨은 YOLO segmentation polygon 형식으로 파싱하여 bbox와 mask GT로 변환하도록 구현했지만, 현재 실행 환경에서는 YOLO `.txt` label 파일이 발견되지 않았다.

| split   | image_id                    | image_path                                                                    | label_path                                                                    | label_status   |   num_objects |
|:--------|:----------------------------|:------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:---------------|--------------:|
| train   | session_01_frame_000001.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000001.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000001.txt | missing_label  |             0 |
| train   | session_01_frame_000002.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000002.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000002.txt | missing_label  |             0 |
| train   | session_01_frame_000003.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000003.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000003.txt | missing_label  |             0 |
| train   | session_01_frame_000004.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000004.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000004.txt | missing_label  |             0 |
| train   | session_01_frame_000005.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000005.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000005.txt | missing_label  |             0 |
| train   | session_01_frame_000006.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000006.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000006.txt | missing_label  |             0 |
| train   | session_01_frame_000007.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000007.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000007.txt | missing_label  |             0 |
| train   | session_01_frame_000008.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000008.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000008.txt | missing_label  |             0 |
| train   | session_01_frame_000009.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000009.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000009.txt | missing_label  |             0 |
| train   | session_01_frame_000010.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000010.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000010.txt | missing_label  |             0 |
| train   | session_01_frame_000011.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000011.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000011.txt | missing_label  |             0 |
| train   | session_01_frame_000012.jpg | /home/ssu/Downloads/yolo_cup_dataset/images/train/session_01_frame_000012.jpg | /home/ssu/Downloads/yolo_cup_dataset/labels/train/session_01_frame_000012.txt | missing_label  |             0 |

표 1-1. Label/weight 진단 요약

| key                    | value                                                                                                                                                                                                                                                                          |
|:-----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset_root           | /home/ssu/Downloads/yolo_cup_dataset                                                                                                                                                                                                                                           |
| label_files_total      | 1                                                                                                                                                                                                                                                                              |
| txt_label_files_total  | 0                                                                                                                                                                                                                                                                              |
| label_extension_counts | .cache:1                                                                                                                                                                                                                                                                       |
| best_pt_count          | 7                                                                                                                                                                                                                                                                              |
| best_pt_candidates     | /home/ssu/Azas/best.pt;/home/ssu/Downloads/best.pt;/home/ssu/Downloads/로봇 데이터/best.pt;/home/ssu/Azas/local_models/best.pt;/home/ssu/Downloads/로봇 데이터/new/best.pt;/home/ssu/Downloads/로봇 데이터/test_predictions/best.pt;/home/ssu/.local/share/Trash/files/best.pt |
| cup_yaml               | /home/ssu/Downloads/yolo_cup_dataset/cup.yaml                                                                                                                                                                                                                                  |

표 1-2. Split별 image-label stem 매칭 진단

| split   | used_split   | image_dir                                         | label_dir                                         |   images |   txt_labels |   matched |   missing_labels |   orphan_labels | sample_missing                                                                                                          | sample_orphan   |
|:--------|:-------------|:--------------------------------------------------|:--------------------------------------------------|---------:|-------------:|----------:|-----------------:|----------------:|:------------------------------------------------------------------------------------------------------------------------|:----------------|
| train   | train        | /home/ssu/Downloads/yolo_cup_dataset/images/train | /home/ssu/Downloads/yolo_cup_dataset/labels/train |      301 |            0 |         0 |              301 |               0 | session_01_frame_000001;session_01_frame_000002;session_01_frame_000003;session_01_frame_000004;session_01_frame_000005 | not measured    |
| val     | val          | /home/ssu/Downloads/yolo_cup_dataset/images/val   | /home/ssu/Downloads/yolo_cup_dataset/labels/val   |       86 |            0 |         0 |               86 |               0 | session_11_frame_000072;session_11_frame_000073;session_11_frame_000074;session_11_frame_000075;session_11_frame_000076 | not measured    |
| test    | test         | /home/ssu/Downloads/yolo_cup_dataset/images/test  | /home/ssu/Downloads/yolo_cup_dataset/labels/test  |       43 |            0 |         0 |               43 |               0 | session_12_frame_000058;session_12_frame_000059;session_12_frame_000060;session_12_frame_000061;session_12_frame_000062 | not measured    |

## 5. 기존 YOLOv8-seg baseline

기존 YOLOv8-seg fine-tuning 방식은 특정 데이터셋에 대해 높은 정밀도를 보일 수 있지만, 새로운 물체나 표현이 추가될 때 재라벨링과 재학습이 필요하다.

`best.pt` 후보는 `/home/ssu` 하위에서 자동 탐색되었고, prediction-only baseline row를 생성했다. 하지만 GT label 파일이 없으므로 Ultralytics validation의 precision/recall/mAP50/mAP50-95는 의미 있는 supervised metric으로 계산할 수 없다.

표 1-3. YOLO best.pt 후보

|   rank | path                                                     |   size_bytes |
|-------:|:---------------------------------------------------------|-------------:|
|      1 | /home/ssu/Azas/best.pt                                   |     23860084 |
|      2 | /home/ssu/Downloads/best.pt                              |     23860084 |
|      3 | /home/ssu/Downloads/로봇 데이터/best.pt                  |      6774260 |
|      4 | /home/ssu/Azas/local_models/best.pt                      |     23860084 |
|      5 | /home/ssu/Downloads/로봇 데이터/new/best.pt              |      6786356 |
|      6 | /home/ssu/Downloads/로봇 데이터/test_predictions/best.pt |      6238698 |
|      7 | /home/ssu/.local/share/Trash/files/best.pt               |      6774260 |

## 6. Foundation model pipeline

반면 GroundingDINO, OWLv2, SAM2 기반 방식은 텍스트 프롬프트와 박스/마스크 프롬프트를 이용하여 별도 학습 없이 객체 후보와 segmentation mask를 생성할 수 있다.

## 7. GroundingDINO + SAM2 방법

GroundingDINO는 `cup`, `lid`, `dispenser` 관련 텍스트 프롬프트로 open-vocabulary box를 생성한다. SAM2는 box prompt 기반 segmentation을 목표로 했으나, 현재 RTX 5080 실행에서는 predictor load 단계에서 CUDA OOM이 발생했다. 따라서 본 실행 결과의 mask row는 실제 SAM2 mask가 아니라 box-only mask fallback임을 분리해서 기록한다.

표 1-4. SAM2 real mask 및 fallback mask 분리

| source             |   sam2_real_masks |   box_fallback_masks |   total_rows |
|:-------------------|------------------:|---------------------:|-------------:|
| GroundingDINO path |                 0 |                  282 |          282 |
| OWLv2 path         |                 0 |                  203 |          203 |

## 8. OWLv2 fallback 방법

OWLv2는 `cup`, `cup lid`, `drink dispenser`, `beverage dispenser`, `plastic cup`, `white lid` 후보 클래스에 대한 zero-shot detector fallback이다.

## 9. Qwen2.5-VL 보조 실험

Qwen2.5-VL은 정량 검출기의 대체재가 아니라, 자연어 지시 기반으로 물체 위치, 색상, 순서 정보를 한 번에 출력할 수 있는지 확인하기 위한 VLM 보조 실험으로 사용하였다.

Qwen2.5-VL 보조 실험은 실행/다운로드를 시도했지만 parsed row는 0개이며 raw 파일 크기는 0 bytes이다. 따라서 성공 결과가 아니라 실패 사례/한계로 기록한다.

## 10. 컵 윗부분 및 조작 후보점 계산 방식

mask를 이용하면 단순 bbox 중심이 아니라 cup top center, cup center, grasp candidate 같은 geometry feature를 계산할 수 있다. 현재 실행에서는 SAM2 OOM 때문에 box-mask fallback이 사용되었으므로, geometry feature는 주로 bbox/fallback mask 기반 perception 후보점으로 해석해야 한다.

## 11. 디스펜서 색상/순서 계산 방식

HSV는 디스펜서 색상 판단에서 단순하지만 안정적인 deterministic baseline이다. box/mask crop 중앙 60%의 median HSV를 계산하고 predefined range로 색상을 분류한다. HSV가 unknown이면 Qwen2.5-VL 보조 색상 결과를 참고하도록 구현했지만, 현재 Qwen parsed row가 없으므로 최종 실행에서는 HSV 중심으로 기록된다.

## 12. 정량 성능 결과

### 12.1 Prediction coverage and runtime summary

GT가 없어도 측정 가능한 항목은 prediction row count, image coverage, confidence distribution, latency, peak VRAM이다. 본 실험의 실행 산출물 row count는 다음과 같다.

| file                                              |   data_rows |
|:--------------------------------------------------|------------:|
| outputs/predictions/foundation_boxes.csv          |         282 |
| outputs/predictions/foundation_masks.csv          |         282 |
| outputs/predictions/owlv2_boxes.csv               |         203 |
| outputs/predictions/owlv2_masks.csv               |         203 |
| outputs/predictions/yolo_baseline_predictions.csv |          12 |
| outputs/predictions/cup_geometry.csv              |          57 |
| outputs/predictions/dispenser_color_order.csv     |         368 |
| outputs/predictions/remove_candidates.csv         |         485 |
| outputs/predictions/qwen25vl_parsed.csv           |           0 |

표 2-1. Prediction row 및 image coverage 요약

| method              |   prediction_rows |   images_with_predictions |   total_images |   coverage_ratio |   mean_conf | notes                                                          |
|:--------------------|------------------:|--------------------------:|---------------:|-----------------:|------------:|:---------------------------------------------------------------|
| groundingdino_sam2  |               282 |                        43 |             43 |           1      |      0.2862 | qualitative count summary; not AP because GT labels are absent |
| owlv2               |               203 |                        43 |             43 |           1      |      0.2638 | qualitative count summary; not AP because GT labels are absent |
| yolov8_seg_baseline |                12 |                        10 |             43 |           0.2326 |      0.5023 | qualitative count summary; not AP because GT labels are absent |

표 2-2. Class-wise prediction count 및 confidence 요약

| method              | class_name   |   count |   mean_conf |
|:--------------------|:-------------|--------:|------------:|
| groundingdino_sam2  | cup          |      57 |      0.2205 |
| groundingdino_sam2  | dispenser    |     173 |      0.2656 |
| groundingdino_sam2  | lid          |      50 |      0.4359 |
| groundingdino_sam2  | other        |       2 |      0.2018 |
| owlv2               | dispenser    |     195 |      0.2653 |
| owlv2               | lid          |       8 |      0.2272 |
| yolov8_seg_baseline | cup          |       6 |      0.3495 |
| yolov8_seg_baseline | lid          |       6 |      0.6552 |

표 2-3. Confidence 분포 요약

| method              |   conf_min |   conf_mean |   conf_median |   conf_max |
|:--------------------|-----------:|------------:|--------------:|-----------:|
| groundingdino_sam2  |     0.2009 |      0.2862 |        0.2536 |     0.5386 |
| owlv2               |     0.201  |      0.2638 |        0.2576 |     0.4363 |
| yolov8_seg_baseline |     0.2698 |      0.5023 |        0.504  |     0.8102 |

표 2-4. 파생 산출물 요약

| artifact              |   rows | nonempty   | notes                                                       |
|:----------------------|-------:|:-----------|:------------------------------------------------------------|
| cup_geometry          |     57 | True       | cup center/top/grasp candidates from masks or bbox fallback |
| dispenser_color_order |    368 | True       | HSV median color and left-to-right order                    |
| remove_candidates     |    485 | True       | perception-only action labels; no robot control             |

표 2-5. 최종 실험 실행 상태

| component              | status                           | evidence                                         |   rows_or_count | notes                                                                                                            |
|:-----------------------|:---------------------------------|:-------------------------------------------------|----------------:|:-----------------------------------------------------------------------------------------------------------------|
| dataset_images         | completed                        | train=301,val=86,test=43 images found            |             430 | Actual dataset root /home/ssu/Downloads/yolo_cup_dataset                                                         |
| gt_labels              | not_available_external_input     | YOLO txt label files found=0                     |               0 | AP/IoU metrics intentionally not fabricated when labels are absent                                               |
| yolo_baseline          | completed_prediction_only_no_gt  | best.pt count=7; prediction_rows=12              |              12 | Weights found, but GT labels absent so validation AP/mAP are not measured; prediction rows are saved             |
| groundingdino_boxes    | completed                        | foundation_boxes.csv                             |             282 | Open-vocabulary detector ran on test split                                                                       |
| sam2_masks_or_fallback | completed_with_box_mask_fallback | foundation_masks.csv rows=282; mask_png=491      |             282 | SAM2 predictor import confirmed earlier; load OOM documented, bbox masks preserve downstream perception pipeline |
| owlv2_fallback         | completed                        | owlv2_boxes.csv/owlv2_masks.csv                  |             203 | Fallback zero-shot detector ran on test split                                                                    |
| qwen25vl_aux           | attempted_no_parsed_rows         | qwen25vl_parsed rows=0                           |               0 | Auxiliary VLM only; model download/load did not finish in final retry                                            |
| cup_geometry           | completed                        | cup_geometry.csv                                 |              57 | Perception geometry candidates only, no actuation                                                                |
| dispenser_color_order  | completed                        | dispenser_color_order.csv                        |             368 | HSV deterministic classifier; manual GT absent so accuracy not measured                                          |
| remove_candidates      | completed                        | remove_candidates.csv                            |             485 | Perception labels only; no robot control code                                                                    |
| report_and_package     | completed                        | report markdown, manifest, audit, submission zip |               1 | Submission artifacts regenerated after final audit                                                               |

표 2-6. 효율성: latency, peak VRAM

| method              | task         | precision    | recall       | f1           | map50        | map50_95     | mean_mask_iou   | cup_center_error_px   | cup_top_error_px   | color_accuracy   | order_exact_match   | latency_ms_mean    | peak_vram_mb   | notes                                                                                      |
|:--------------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:----------------|:----------------------|:-------------------|:-----------------|:--------------------|:-------------------|:---------------|:-------------------------------------------------------------------------------------------|
| groundingdino_sam2  | detection    | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | 4389.1802268645215 | 0.0            | not measured because YOLO GT label files are absent                                        |
| owlv2               | detection    | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | 2022.831033161736  | 0.0            | not measured because YOLO GT label files are absent                                        |
| yolov8_seg_baseline | detection    | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | 42.78462842041843  | 0.0            | not measured because YOLO GT label files are absent                                        |
| groundingdino_sam2  | segmentation | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | 4389.1802268645215 | 0.0            | not measured because YOLO GT label files are absent; box-mask fallback if SAM2 unavailable |
| owlv2               | segmentation | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | 2022.831033161736  | 0.0            | not measured because YOLO GT label files are absent; box-mask fallback if SAM2 unavailable |
| yolov8_seg_baseline | segmentation | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | 42.78462842041843  | 0.0            | not measured because YOLO GT label files are absent; box-mask fallback if SAM2 unavailable |
| all                 | geometry     | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | not measured       | not measured   | GT keypoints unavailable; geometry features generated but error not measured               |
| hsv_plus_vlm_aux    | dispenser    | not measured | not measured | not measured | not measured | not measured | not measured    | not measured          | not measured       | not measured     | not measured        | not measured       | not measured   | HSV deterministic baseline, Qwen2.5-VL optional color aid                                  |

### 12.2 Manual GT subset evaluation, if available

현재 공식 GT label 파일이 없기 때문에 `outputs/manual_eval/manual_gt_bboxes.csv` 템플릿과 `outputs/manual_eval/preview/` 이미지를 생성했다. 사용자가 10~20장에 대해 `image_id,class_name,x1,y1,x2,y2`를 채우면 `scripts/13_evaluate_manual_gt.py`가 foundation, OWLv2, YOLO prediction을 IoU 0.5 및 0.5:0.95 sweep으로 비교한다. 수동 GT가 비어 있으면 아래 표처럼 not measured로 유지한다.

| method           |   num_gt |   num_pred | iou_threshold   | tp           | fp           | fn           | precision    | recall       | f1           | ap50         | ap50_95      | notes                                                                                       |
|:-----------------|---------:|-----------:|:----------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:--------------------------------------------------------------------------------------------|
| manual_gt_subset |        0 |          0 | not measured    | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | manual_gt_bboxes.csv has no filled GT rows; fill image_id,class_name,x1,y1,x2,y2 then rerun |

### 12.3 GT-based AP/IoU evaluation unavailable because labels are absent

표 2. Detection 성능

not measured

표 3. Segmentation 성능

not measured

표 4. 컵 윗부분/조작 후보점 성능

| method                          | image_id                    | cup_center_error_px   | cup_top_error_px   | grasp_candidate_error_px   | notes                                                 |
|:--------------------------------|:----------------------------|:----------------------|:-------------------|:---------------------------|:------------------------------------------------------|
| groundingdino_sam2_box_fallback | session_12_frame_000058.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000058.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000058.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000058.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000058.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000059.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000059.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000059.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000061.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000062.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000065.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000065.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000065.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000066.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000066.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000066.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000067.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000067.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000068.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |
| groundingdino_sam2_box_fallback | session_12_frame_000068.jpg | not measured          | not measured       | not measured               | GT keypoints are not available in YOLO polygon labels |

표 5. 디스펜서 색상/순서 성능

| method           | color_accuracy   | order_exact_match   | notes                                         |
|:-----------------|:-----------------|:--------------------|:----------------------------------------------|
| hsv_plus_vlm_aux | not measured     | not measured        | manual_color_labels.csv required for accuracy |

## 13. 정성 시각화 결과

대표 report asset:

![sample](outputs/report_assets/sample_01_foundation_samples.png)
![sample](outputs/report_assets/sample_02_foundation_samples.png)
![sample](outputs/report_assets/sample_03_foundation_samples.png)
![sample](outputs/report_assets/sample_04_foundation_samples.png)
![sample](outputs/report_assets/sample_04_yolo_samples.png)
![sample](outputs/report_assets/sample_05_foundation_samples.png)

Contact sheet:

![dispenser_contact_sheet](outputs/report_assets/contact_sheets/dispenser_contact_sheet.png)
![foundation_contact_sheet](outputs/report_assets/contact_sheets/foundation_contact_sheet.png)
![gt_contact_sheet](outputs/report_assets/contact_sheets/gt_contact_sheet.png)
![remove_candidate_contact_sheet](outputs/report_assets/contact_sheets/remove_candidate_contact_sheet.png)
![yolo_contact_sheet](outputs/report_assets/contact_sheets/yolo_contact_sheet.png)

Manual GT annotation preview는 `outputs/manual_eval/preview/`에 저장되어 있다.

## 14. 실행 상태 및 실패 사례 분석

최종 실행 상태는 다음과 같다. GroundingDINO는 test 43장 전체에 대해 282개 prediction row를 생성했고, OWLv2 fallback은 203개 prediction row를 생성했다. YOLOv8-seg `best.pt` 후보도 발견되어 prediction-only baseline 12개 row를 생성했다. 컵 geometry는 57개, 디스펜서 색상/순서는 368개, remove_candidate perception label은 485개 row가 생성되었다.

제한 사항도 명확하다. 현재 데이터셋의 `labels/` 폴더에는 YOLO `.txt` label 파일이 없어 AP, recall, mask IoU 같은 GT 기반 metric은 계산하지 않았다. YOLO baseline weight는 발견되었지만 GT가 없으므로 validation mAP는 성능 지표로 해석하지 않는다. SAM2는 `--no-build-isolation --no-deps` 방식으로 설치하고 predictor import까지 확인했지만, 실제 predictor load 단계에서 CUDA OOM이 발생하여 box-only mask fallback을 사용하였다. Qwen2.5-VL 보조 실험은 실행/다운로드를 시도했지만 parsed row는 0개이며 raw 파일 크기는 0 bytes이다. 따라서 성공 결과가 아니라 실패 사례/한계로 기록한다.

Foundation model은 프롬프트 표현, threshold, 조명, 투명 컵 경계, 흰색 뚜껑과 배경의 대비에 민감하다. SAM2가 설치되지 않거나 메모리가 부족하면 box-only fallback이 사용되어 mask IoU가 낮아질 수 있다.

## 15. 한계점

본 결과는 완전한 supervised benchmark가 아니다. GT label 파일과 manual color label이 없는 경우 detection AP, segmentation IoU, color accuracy, order exact match accuracy는 not measured로 기록된다. YOLO polygon GT에는 cup top keypoint가 없으므로 geometry error는 별도 keypoint GT가 없으면 정량 측정하지 않는다. 성능 측정을 강화하려면 최소 10~20장 manual GT subset을 채운 뒤 manual detection metric을 추가해야 한다.

## 16. 결론

본 실험에서는 RTX 5080 환경에서 GroundingDINO와 OWLv2 기반 zero-shot object detection을 실행하고, 컵/뚜껑/디스펜서 후보 탐지, 컵 조작 후보점 계산, 디스펜서 색상/순서 추정을 수행하였다. 다만 현재 데이터셋에는 GT label 파일이 없어 AP, recall, mask IoU 같은 GT 기반 정량 지표는 계산하지 않았다. YOLO baseline weight는 발견되어 prediction-only baseline을 생성했지만, GT 부재 때문에 supervised mAP benchmark는 수행하지 않았다. SAM2는 predictor load 단계에서 CUDA OOM이 발생하여 box-only mask fallback을 사용하였다. 따라서 본 결과는 완전한 supervised benchmark가 아니라, foundation model 기반 perception pipeline의 실행 가능성, prediction coverage, latency, downstream feature 생성 가능성을 확인한 실험으로 해석해야 한다.

YOLOv8-seg는 fine-tuned supervised baseline이며, GroundingDINO + SAM2는 별도 재학습 없이 텍스트 프롬프트로 cup/lid/dispenser를 찾고 mask를 생성하는 foundation model pipeline이다. 단, 이번 실행의 segmentation 결과는 실제 SAM2 mask가 아니라 box-mask fallback이므로 보고서와 CSV에서 이를 분리해 표기하였다. OWLv2는 zero-shot object detection fallback이고, Qwen2.5-VL은 bbox/color/order JSON 출력 가능성을 보는 VLM 보조 실험이다. 실제 로봇 제어는 하지 않았고 perception 결과만 만들었다.
