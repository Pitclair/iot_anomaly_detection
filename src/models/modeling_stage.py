"""
Modeling stage: fit Dirichlet alpha parameters ("Normal" profile) using Minka fixed-point.
"""
from typing import Optional
import numpy as np
import json
from pathlib import Path

from algorithms.dirichlet import fixed_point_dirichlet


def run_modeling(cfg: dict, dataset: str = 'camera_5') -> None:
    """Run modeling stage: load counts, estimate alpha, and persist model.
    Currently expects counts available at cfg['ingest']['path'] as a CSV of raw packets.
    """
    ingest = cfg.get('ingest', {})
    data_path = ingest.get('path')
    # For scaffold: assume processed counts are available as a CSV with one row per window and columns for categories
    # Try to load data_path; if missing, create a small synthetic dataset
    p = Path(data_path)
    if not p.exists():
        # synth counts: 100 windows, K categories
        K = cfg.get('categories_k', 4)
        counts = np.random.poisson(lam=5.0, size=(100, K)).astype(int)
    else:
        import pandas as pd
        df = pd.read_csv(p)
        counts = df.values.astype(int)

    alpha_init = None
    tol = cfg.get('tolerance_delta', 1e-9)
    alpha, info = fixed_point_dirichlet(counts, alpha_init=alpha_init, tol=tol)

    out_path = Path(cfg.get('model', {}).get('persist_path', 'data/processed/model_alpha.json'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {'alpha': alpha.tolist(), 'info': info}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Model saved to {out_path} (converged={info.get('converged')})")

