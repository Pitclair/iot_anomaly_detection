"""
Forecasting stage: generate horizon forecasts (24h) and compute Brier Score and JSD against actuals.
"""
from typing import Optional
import numpy as np
from pathlib import Path

from lm_idnet.artifacts import load_artifact
from lm_idnet.evaluation.metrics import brier_score, js_divergence


def run_forecasting(cfg: dict, dataset: str = 'camera_5') -> None:
    model_path = Path(cfg["outputs"]["model_path"])
    model = load_artifact(model_path, expected_type="model")
    alpha = np.array(model.alpha, dtype=float)

    # expected probabilities from Dirichlet mean
    probs = alpha / alpha.sum()
    horizon_hours = cfg.get('forecast', {}).get('horizon_hours', 24)

    # synthetic actual: sample from multinomial with same total counts per hour (for scaffold)
    total_per_hour = 100
    rng = np.random.default_rng(0)
    actual = rng.multinomial(total_per_hour, probs, size=horizon_hours)  # shape (H, K)

    # forecast: use expected counts = total_per_hour * probs
    forecast = np.tile((total_per_hour * probs), (horizon_hours, 1))

    # compute metrics per hour and average
    bs = brier_score(actual / total_per_hour, forecast / total_per_hour)
    js = js_divergence(actual / total_per_hour + 1e-12, forecast / total_per_hour + 1e-12)

    print(f"Forecasting: horizon={horizon_hours}h, BrierScore={bs:.6f}, JSD={js:.6f}")
