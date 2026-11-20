# AI-03: Fall Detection from Phone and Wearable Sensors

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime%20v1.16+-brightgreen.svg)](https://onnxruntime.ai/)
[![Tests](https://img.shields.io/badge/tests-98%20passed-success.svg)](./AI-03-Fall-Detection/tests)

**Organization:** TheCarePal  
**Project:** AI-03 — Fall Detection from Phone and Wearable Sensors  
**Author:** Farhan Machchhar  

---

## 📌 Project Overview

**CareGuard** is an end-to-end, machine-learning-powered fall detection and telecare assistance system built using smartphone motion sensors (**RIGHTPOCKET**) and wrist-worn smartwatch sensors (**WRIST**) from the public **UMAFall** dataset (746 recordings across 17 subjects).

### Key Features
- **Dual-Sensor Inertial Telemetry**: Real-time evaluation of 3-axis accelerometer and gyroscope motion streams.
- **5-Row Autonomous Fallback Matrix**: Seamless degradation between full dual-sensor fusion, phone-only fallback, wearable-only fallback, and sensor offline detection.
- **Embedded ONNX Inference**: High-performance scikit-learn Random Forest and MLP classifiers serialized to ONNX format.
- **Non-Blocking 30-Second Emergency Alert**: Real-time alert countdown with one-click user cancellation and simulated caregiver notification dispatch upon expiry.
- **Interactive Streamlit Dashboard**: Real-time kinematic visualization, scenario presets, confusion matrices, and CSV audit log downloads.

---

## 📂 Repository Structure

```text
Fall-Detection/
├── app.py                                # Root entry point for Streamlit Cloud
├── requirements.txt                      # Project dependencies
├── Dataset/                              # 746 raw UMAFall CSV sensor recordings
├── AI-03 - Fall Detection...docx         # Project assignment brief
├── AI-03-Fall-Detection/                 # Main application directory
│   ├── app.py                            # Streamlit dashboard implementation
│   ├── requirements.txt                  # Application requirements
│   ├── README.md                         # Detailed technical documentation
│   ├── PROJECT_STATUS.md                 # Complete verification & milestone report
│   ├── DEMO_SCRIPT.md                    # 4-minute evaluation demonstration guide
│   ├── src/                              # Production source modules
│   │   ├── data_loader.py                # CSV ingestion & dynamic header parsing
│   │   ├── preprocessing.py              # 2.0s sliding-window segmentation
│   │   ├── feature_extraction.py         # 66 kinetic feature extraction
│   │   ├── train.py                      # Participant-level model training
│   │   ├── evaluate.py                   # Performance benchmarking & metrics
│   │   ├── inference.py                  # ONNX Runtime inference wrapper
│   │   ├── sensor_simulator.py           # BLE link & on-wrist replay simulator
│   │   ├── sensor_fusion.py              # 5-row decision fusion engine
│   │   └── alert_manager.py              # 30s countdown state machine
│   ├── models/                           # Serialized ONNX & pickle models
│   ├── reports/                          # Confusion matrices & evaluation metrics
│   ├── data/                             # Processed feature datasets
│   ├── tests/                            # Automated pytest suite (98 tests)
│   └── scripts/                          # Automated training & pipeline runners
```

---

## 🚀 Quick Start (Local Run)

### 1. Clone the Repository
```bash
git clone https://github.com/Farhan-Machchhar/Fall-Detection.git
cd Fall-Detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Or directly from the application folder:
```bash
cd AI-03-Fall-Detection
streamlit run app.py
```

### 4. Run Automated Test Suite
```bash
pytest AI-03-Fall-Detection/tests -v
```

---

## ☁️ How to Host on Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **"New app"**.
3. Select your GitHub repository:
   - **Repository:** `Farhan-Machchhar/Fall-Detection`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py` (or `AI-03-Fall-Detection/app.py`)
4. Click **"Deploy!"**.
5. Streamlit Cloud will install dependencies from `requirements.txt` and launch the CareGuard dashboard.

---

## 📊 Measured Model Performance

Evaluated strictly on independent held-out participants (Subjects 12–17, 0% subject leakage):

| Model Architecture | Placement | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | False Alarms / Hour |
|---|---|---|---|---|---|---|---|
| **Random Forest (Primary)** | **Smartphone (RIGHTPOCKET)** | **93.29%** | **92.84%** | **93.40%** | **76.59%** | **83.94%** | **2.31 / hr** |
| **Multilayer Perceptron** | **Smartphone (RIGHTPOCKET)** | **94.12%** | **89.26%** | **95.26%** | **81.41%** | **85.15%** | **2.31 / hr** |
| **Random Forest (Primary)** | **Smartwatch (WRIST)** | **88.87%** | **87.33%** | **89.23%** | **65.36%** | **74.76%** | **2.31 / hr** |
| **Multilayer Perceptron** | **Smartwatch (WRIST)** | **89.76%** | **82.92%** | **91.35%** | **69.04%** | **75.34%** | **2.31 / hr** |

---

## ⚖️ Clinical Disclaimer
**CareGuard is an experimental assistance tool and algorithmic research demonstration.** It has not been cleared, approved, or evaluated by the FDA, CE, or CDSCO. It does not provide medical diagnoses or replace emergency medical response services.
