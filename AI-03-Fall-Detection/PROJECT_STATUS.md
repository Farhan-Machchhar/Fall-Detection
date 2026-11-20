# AI-03 Fall Detection — Project Status

> **Status:** All 16 Stages Completed & Verified ✅  
> **Last updated:** 2026-11-17  
> **Automated Tests:** 98 Passed, 0 Failed (100% Pass Rate)

## Project Overview
Autonomous dual-device elderly fall-detection system using smartphone motion sensors (**RIGHTPOCKET**) and smartwatch/wrist wearable sensors (**WRIST**) on the public **UMAFall** dataset (746 CSV recordings across 17 subjects).

Features:
- Primary machine learning models: **Random Forest Classifiers**
- Comparison machine learning models: **Multilayer Perceptrons (MLP)**
- Edge runtime serialization: **ONNX format** (`skl2onnx`, `onnxruntime`)
- Sensor fusion: **5-row decision matrix** with dynamic weights and conflict resolution
- Real-time simulation: **BLE connection / on-wrist status simulation** with dataset replay
- Emergency alerting: **30-second cancellation countdown** with simulated caregiver notification
- User Interface: Full **Streamlit dashboard** (`app.py`) with 6 distinct operational views

---

## Stage Progress Tracker

| # | Stage | Status | Deliverables & Verification Notes |
|---|---|---|---|
| 1 | Dataset inspection & structure validation | ✅ Done | Inspected 746 CSVs; confirmed metadata format, 200 Hz sampling, 5 sensor placements |
| 2 | Data loading pipeline (`src/data_loader.py`) | ✅ Done | Dynamic header detection, metadata extraction, graceful malformed-file handling |
| 3 | Preprocessing & windowing (`src/preprocessing.py`) | ✅ Done | 2.0 s windows, 1.0 s stride (50% overlap), timestamp-gap segmentation |
| 4 | Feature engineering (`src/feature_extraction.py`) | ✅ Done | 66 features per window (33 accel + 33 gyro): mean, std, min, max, rms, energy, variance, median, IQR, magnitude, jerk |
| 5 | Dataset generation (`data/`) | ✅ Done | `phone_features.csv` (7.7 MB, 5,594 windows) and `wrist_features.csv` (7.7 MB, 5,594 windows) |
| 6 | Participant-level train/test split | ✅ Done | Zero leakage across participants (Subjects 1–11 train, 12–17 test) |
| 7 | Model training (`src/train.py`) | ✅ Done | Trained 4 models: Phone RF, Phone MLP, Wrist RF, Wrist MLP |
| 8 | ONNX export & numerical verification (`models/`) | ✅ Done | Exported `phone_model.onnx` & `wearable_model.onnx`; sklearn vs ONNX diff = 0.000000 |
| 9 | Rigorous evaluation (`src/evaluate.py`, `reports/`) | ✅ Done | Accuracy, Recall, Specificity, Precision, F1, False Alarms/hr, confusion matrices, comparison chart |
| 10 | Inference runtime engine (`src/inference.py`) | ✅ Done | High-performance ONNX Runtime wrapper with automatic fallback to pickle |
| 11 | Wearable sensor simulator (`src/sensor_simulator.py`) | ✅ Done | BLE link toggle, on-wrist detection, paired stream replay, scenario preset loader |
| 12 | Decision-table sensor fusion (`src/sensor_fusion.py`) | ✅ Done | Complete 5-row decision matrix with conflict resolution and configurable weights |
| 13 | Emergency alert manager (`src/alert_manager.py`) | ✅ Done | Non-blocking 30 s countdown, user cancellation, duplicate suppression (60 s cooldown) |
| 14 | Automated test suite (`tests/`) | ✅ Done | 98 automated unit and end-to-end requirement tests passing in 7.4 seconds |
| 15 | Interactive Streamlit application (`app.py`) | ✅ Done | 6 responsive sections, live sensor stream charts, alert countdown, CSV log export |
| 16 | Documentation & Demo Script (`README.md`, `DEMO_SCRIPT.md`) | ✅ Done | Complete documentation with exact commands and 4-minute interview presentation script |

---

## Actual Measured Model Performance

Evaluated on independent held-out participants (no participant overlap, no window leakage):

| Model Architecture | Placement | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | False Alarms / Hour |
|---|---|---|---|---|---|---|---|
| **Random Forest (Primary)** | **Smartphone (RIGHTPOCKET)** | **93.29%** | **92.84%** | **93.40%** | **76.59%** | **83.94%** | **2.31 / hr** |
| Multilayer Perceptron (MLP) | Smartphone (RIGHTPOCKET) | 94.07% | 89.26% | 95.19% | 81.20% | 85.04% | 2.31 / hr |
| **Random Forest (Primary)** | **Wrist Wearable (WRIST)** | **88.87%** | **87.33%** | **89.23%** | **65.36%** | **74.76%** | **2.31 / hr** |
| Multilayer Perceptron (MLP) | Wrist Wearable (WRIST) | 89.76% | 82.92% | 91.35% | 69.04% | 75.34% | 2.31 / hr |

*Note: False alarms per hour were computed by grouping consecutive positive windows with a 5-second event cooldown over normal continuous ADL recordings.*

---

## Automated Verification Matrix (19 Mandatory Items)

| Item | Requirement Description | Verification Method | Status |
|---|---|---|---|
| 1 | Loading a fall recording | `test_item_01_loading_fall_recording` | ✅ Passed |
| 2 | Loading a normal ADL recording | `test_item_02_loading_adl_recording` | ✅ Passed |
| 3 | Extracting phone features | `test_item_03_extracting_phone_features` | ✅ Passed |
| 4 | Extracting wrist features | `test_item_04_extracting_wrist_features` | ✅ Passed |
| 5 | Processing multiple CSV files | `test_item_05_processing_multiple_csv_files` | ✅ Passed |
| 6 | Handling missing sensor streams | `test_item_06_handling_missing_sensor_streams` | ✅ Passed |
| 7 | Training both models | `test_item_07_trained_models_exist` | ✅ Passed |
| 8 | Generating evaluation metrics | `test_item_08_evaluation_metrics_generated` | ✅ Passed |
| 9 | ONNX inference execution | `test_item_09_onnx_inference` | ✅ Passed |
| 10 | Phone-only detection | `test_item_10_phone_only_detection` | ✅ Passed |
| 11 | Wearable-only detection | `test_item_11_wearable_only_detection` | ✅ Passed |
| 12 | Both-device fusion | `test_item_12_both_device_fusion` | ✅ Passed |
| 13 | Wearable disconnection fallback | `test_item_13_wearable_disconnection_fallback` | ✅ Passed |
| 14 | Wearable not-worn fallback | `test_item_14_wearable_not_worn_fallback` | ✅ Passed |
| 15 | Sensor unavailable state | `test_item_15_sensor_unavailable_state` | ✅ Passed |
| 16 | Fall alert countdown | `test_item_16_fall_alert_countdown` | ✅ Passed |
| 17 | Alert cancellation | `test_item_17_alert_cancellation` | ✅ Passed |
| 18 | Simulated caregiver notification | `test_item_18_simulated_caregiver_notification` | ✅ Passed |
| 19 | Application startup | `test_item_19_application_startup` | ✅ Passed |

---

## Generated Artifacts

- **Trained Models:**
  - `models/phone_model.onnx` (5.75 MB)
  - `models/wearable_model.onnx` (6.85 MB)
  - `models/phone_rf_model.pkl` (11.89 MB)
  - `models/wrist_rf_model.pkl` (14.12 MB)
  - `models/phone_mlp.onnx` (69 KB)
  - `models/wrist_mlp.onnx` (69 KB)
  - `models/model_metadata.json` (5.5 KB)
- **Features & Predictions:**
  - `data/phone_features.csv` (7.70 MB)
  - `data/wrist_features.csv` (7.70 MB)
  - `data/predictions/phone_random_forest_predictions.csv`
  - `data/predictions/wrist_random_forest_predictions.csv`
- **Reports & Visualisations:**
  - `reports/evaluation_results.json`
  - `reports/phone_confusion_matrix.png`
  - `reports/wearable_confusion_matrix.png`
  - `reports/model_comparison.png`
  - `reports/confusion_matrix_phone_mlp.png`
  - `reports/confusion_matrix_wrist_mlp.png`
