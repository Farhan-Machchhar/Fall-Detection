"""
AI-03 Fall Detection — Source Package
======================================
This package contains all modules for the fall detection pipeline:

Modules
-------
data_loader        : Parse UMAFall CSV files into pandas DataFrames.
preprocessing      : Sliding-window segmentation of sensor streams.
feature_extraction : Time-domain feature engineering per window.
train              : Model training pipeline (RandomForest + MLP).
evaluate           : Metrics, confusion matrix, and report generation.
inference          : ONNX model loading and real-time prediction.
sensor_simulator   : Simulate a wearable sensor stream from recorded data.
sensor_fusion      : Combine phone + wearable predictions.
alert_manager      : Fall-alert state machine (countdown / cooldown).
"""

__version__ = "0.1.0"
__author__ = "AI-03 Team"
