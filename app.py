"""
Streamlit Cloud Entry Point
===========================
CareGuard — AI-Powered Fall Detection from Phone and Wearable Sensors.
Assignment: AI-03 (TheCarePal)

This file acts as the root entry point for hosting on Streamlit Community Cloud.
It dynamically routes execution to the main application in `AI-03-Fall-Detection/app.py`.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
APP_DIR = ROOT_DIR / "AI-03-Fall-Detection"
SRC_DIR = APP_DIR / "src"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(SRC_DIR))

# Ensure DATASET_DIR points to root Dataset folder if available
if (ROOT_DIR / "Dataset").exists() and not os.getenv("DATASET_DIR"):
    os.environ["DATASET_DIR"] = str(ROOT_DIR / "Dataset")

# Execute main application script
app_script = APP_DIR / "app.py"
with open(app_script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, str(app_script), "exec"))
