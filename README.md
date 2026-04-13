IoT Anomaly Detection Baseline (Python 3.11)

This repo provides a scaffold for modeling and forecasting IoT network traffic using Dirichlet-based methods.

Structure:
- src/: Python package modules (processing, algorithms, models, evaluation)
- configs/config.json: default configuration
- docker-compose.yml: run the app in python:3.11-slim without a Dockerfile

Usage (local):

python main.py model --config configs/config.json
python main.py forecast --config configs/config.json

