# Assignment 2: Anomaly Detection Using LSTM Autoencoders

## Overview
This repository contains the implementation and analysis for anomaly detection using LSTM-based autoencoder models. The project includes two different LSTM autoencoder architectures and one baseline model for comparison.

## Dataset
NASA Anomaly Detection Dataset (SMAP & MSL)
- SMAP (Soil Moisture Active Passive satellite) telemetry data
- MSL (Mars Science Laboratory rover) telemetry data
- Contains labeled anomalies for supervised evaluation
- Time-series telemetry data with various sensors and operational parameters

## Directory Structure
- notebooks/: Jupyter notebooks containing all implementations
- saved_models/: Saved weights for trained models
- docs/: Documentation files with detailed explanations
- figures/: Visualizations and plots

## Implementation Details
- Two LSTM autoencoder variants for anomaly detection
- One baseline model for comparison
- Evaluation using reconstruction error, precision, recall, F1-score, ROC-AUC
- Comparative analysis of all three approaches



