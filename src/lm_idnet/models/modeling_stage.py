"""
Modeling stage: fit Dirichlet alpha parameters ("Normal" profile) using Minka fixed-point.
"""
from typing import Optional
import numpy as np
import json
from pathlib import Path

from lm_idnet.algorithms.dirichlet import fixed_point_dirichlet


def run_modeling(data_path: Path, categories_k: int, tolerance_delta: float, model_out_path: str) -> None:
    """Run modeling stage: load counts, estimate alpha, and persist model.
    data_path: Path object to processed counts CSV
    categories_k: number of categories
    tolerance_delta: convergence tolerance for Dirichlet fitting
    model_out_path: path to save the model
    """
    # p = data_path
    # if not p.exists():
    #     # synth counts: 100 windows, K categories
    #     counts = np.random.poisson(lam=5.0, size=(100, categories_k)).astype(int)
    # else:
    #     import pandas as pd
    #     df = pd.read_csv(p)
    #     counts = df.values.astype(int)
    #
    # alpha_init = None
    # tol = tolerance_delta
    # alpha, info = fixed_point_dirichlet(counts, alpha_init=alpha_init, tol=tol)
    #
    # out_path = Path(model_out_path)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # out = {'alpha': alpha.tolist(), 'info': info}
    # out_path.write_text(json.dumps(out, indent=2))
    # print(f"Model saved to {out_path} (converged={info.get('converged')})")
