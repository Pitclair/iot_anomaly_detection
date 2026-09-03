"""Reproduce the professor-style aggregate convergence experiment."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import scipy
from scipy.special import digamma, rel_entr

from lm_idnet.algorithms.log_likelihood import LmLogLikelihood, ScipyLogLikelihood
from lm_idnet.processing.storage import load_processed_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/D-LinkDayCam5"
OUTPUT = ROOT / "reports/D-LinkDayCam5/lm_professor_experiment.json"
TRAIN_DATES = ("2020-10-08", "2020-10-09", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16")
TEST_DATES = ("2020-10-19", "2020-10-21", "2020-10-22")
FRACTIONS = (0.10, 0.15, 0.20, 0.25)
SEED = 1004
REPEATS = 3
TOLERANCE = 2e-9
MAX_ITERATIONS = 100_000


def load_day(date: str) -> np.ndarray:
    path = next(DATA_DIR.glob(f"*{date}.json"))
    dataset = load_processed_dataset(path)
    counts = np.asarray(
        [window.counts for window in dataset.windows if window.counts is not None],
        dtype=np.float64,
    )
    # The professor's loader adds one to every category when any cell is zero.
    counts[(counts == 0).any(axis=1)] += 1
    return counts


def professor_initial_alpha(counts: np.ndarray) -> np.ndarray:
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    first_moment = probabilities.mean(axis=0)
    second_moment = np.square(probabilities).mean(axis=0)
    scale = (first_moment[0] - second_moment[0]) / (
        second_moment[0] - first_moment[0] ** 2
    )
    alpha = scale * first_moment
    if not np.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError("professor initial-alpha estimate is invalid")
    return alpha


def fit(counts: np.ndarray, backend) -> dict[str, object]:
    """Match the professor's row-wise update and aggregate likelihood check."""
    aggregate = counts.sum(axis=0, keepdims=True)
    alpha = professor_initial_alpha(counts)
    likelihood_seconds = 0.0
    started = perf_counter()
    converged = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        concentration = alpha.sum()
        numerator = np.sum(digamma(counts + alpha) - digamma(alpha), axis=0)
        denominator = np.sum(
            digamma(counts.sum(axis=1) + concentration) - digamma(concentration)
        )
        next_alpha = alpha * numerator / denominator

        likelihood_started = perf_counter()
        old_likelihood = backend.calculate(aggregate, alpha)
        new_likelihood = backend.calculate(aggregate, next_alpha)
        likelihood_seconds += perf_counter() - likelihood_started
        alpha = next_alpha
        if abs(new_likelihood - old_likelihood) < TOLERANCE:
            converged = True
            break

    return {
        "alpha": alpha.tolist(),
        "converged": converged,
        "iterations": iteration,
        "likelihood_seconds": likelihood_seconds,
        "wall_seconds": perf_counter() - started,
    }


def distances(actual: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    difference = actual - estimate
    return {
        "tvd": float(0.5 * np.abs(difference).sum()),
        "kl_divergence": float(rel_entr(actual, estimate).sum()),
        "euclidean": float(np.linalg.norm(difference)),
        "mse": float(np.square(difference).mean()),
    }


def forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    midpoint = 0.5 * (actual + forecast)
    return {
        "brier_score": float(np.square(actual - forecast).sum()),
        "js_divergence": float(
            0.5
            * (rel_entr(actual, midpoint).sum() + rel_entr(forecast, midpoint).sum())
        ),
    }


def main() -> None:
    days = {date: load_day(date) for date in TRAIN_DATES + TEST_DATES}
    backends = {"scipy": ScipyLogLikelihood(), "lm": LmLogLikelihood(6)}
    records: list[dict[str, object]] = []

    # Warm both native paths before recording timings.
    warm_counts = days[TRAIN_DATES[0]][:10]
    warm_alpha = professor_initial_alpha(warm_counts)
    for backend in backends.values():
        backend.calculate(warm_counts.sum(axis=0, keepdims=True), warm_alpha)

    for fraction_index, fraction in enumerate(FRACTIONS):
        for repeat in range(REPEATS):
            for day_index, date in enumerate(TRAIN_DATES):
                full_counts = days[date]
                rng = np.random.default_rng(
                    np.random.SeedSequence([SEED, fraction_index, repeat, day_index])
                )
                sample_size = int(fraction * len(full_counts))
                sample = full_counts[
                    rng.choice(len(full_counts), size=sample_size, replace=False)
                ]
                actual = full_counts.sum(axis=0)
                actual /= actual.sum()
                # Alternate order to avoid consistently favoring the second backend.
                order = ("scipy", "lm") if repeat % 2 == 0 else ("lm", "scipy")
                for name in order:
                    result = fit(sample, backends[name])
                    alpha = np.asarray(result.pop("alpha"), dtype=np.float64)
                    records.append(
                        {
                            "backend": name,
                            "date": date,
                            "fraction": fraction,
                            "repeat": repeat,
                            "sample_size": sample_size,
                            "alpha": alpha.tolist(),
                            "psi": float(1.0 / alpha.sum()),
                            "fit": result,
                            "goodness_of_fit": distances(actual, alpha / alpha.sum()),
                        }
                    )

    summaries: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        for backend in backends:
            selected = [
                record
                for record in records
                if record["fraction"] == fraction and record["backend"] == backend
            ]
            totals = []
            for repeat in range(REPEATS):
                repeated = [record for record in selected if record["repeat"] == repeat]
                totals.append(
                    {
                        "repeat": repeat,
                        "wall_seconds": sum(record["fit"]["wall_seconds"] for record in repeated),
                        "likelihood_seconds": sum(
                            record["fit"]["likelihood_seconds"] for record in repeated
                        ),
                        "iterations": sum(record["fit"]["iterations"] for record in repeated),
                    }
                )
            summaries.append(
                {
                    "backend": backend,
                    "fraction": fraction,
                    "seven_day_totals": totals,
                    "mean_wall_seconds": mean(total["wall_seconds"] for total in totals),
                    "mean_likelihood_seconds": mean(
                        total["likelihood_seconds"] for total in totals
                    ),
                    "mean_iterations": mean(total["iterations"] for total in totals),
                    "converged_fits": sum(record["fit"]["converged"] for record in selected),
                    "total_fits": len(selected),
                    "mean_goodness_of_fit": {
                        metric: mean(record["goodness_of_fit"][metric] for record in selected)
                        for metric in ("tvd", "kl_divergence", "euclidean", "mse")
                    },
                }
            )

    forecasts: list[dict[str, object]] = []
    for repeat in range(REPEATS):
        for backend in backends:
            fitted = [
                record
                for record in records
                if record["fraction"] == 0.25
                and record["repeat"] == repeat
                and record["backend"] == backend
            ]
            fitted.sort(key=lambda record: TRAIN_DATES.index(record["date"]))
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 99, repeat]))
            samples = np.concatenate(
                [rng.dirichlet(np.asarray(record["alpha"]), size=1000) for record in fitted]
            )
            forecast = samples.mean(axis=0)
            for date in TEST_DATES:
                actual = days[date].sum(axis=0)
                actual /= actual.sum()
                forecasts.append(
                    {
                        "backend": backend,
                        "date": date,
                        "repeat": repeat,
                        "forecast": forecast.tolist(),
                        **forecast_metrics(actual, forecast),
                    }
                )

    paired = []
    for scipy_record in (record for record in records if record["backend"] == "scipy"):
        lm_record = next(
            record
            for record in records
            if record["backend"] == "lm"
            and record["date"] == scipy_record["date"]
            and record["fraction"] == scipy_record["fraction"]
            and record["repeat"] == scipy_record["repeat"]
        )
        scipy_alpha = np.asarray(scipy_record["alpha"])
        lm_alpha = np.asarray(lm_record["alpha"])
        paired.append(
            {
                "maximum_relative_alpha_difference": float(
                    np.max(np.abs((lm_alpha - scipy_alpha) / scipy_alpha))
                ),
                "iteration_difference": lm_record["fit"]["iterations"]
                - scipy_record["fit"]["iterations"],
            }
        )

    report = {
        "report_type": "lm_professor_architecture_experiment",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "method": {
            "training_dates": TRAIN_DATES,
            "testing_dates": TEST_DATES,
            "fractions": FRACTIONS,
            "repeats": REPEATS,
            "seed": SEED,
            "tolerance": TOLERANCE,
            "precision_digits": 6,
            "likelihood_semantics": "aggregate counts for convergence only",
            "update_semantics": "independent rows",
            "likelihood_calls_per_iteration": 2,
            "forecast_samples_per_daily_model": 1000,
        },
        "summary": summaries,
        "paired_backend_comparison": {
            "fits": len(paired),
            "maximum_relative_alpha_difference": max(
                item["maximum_relative_alpha_difference"] for item in paired
            ),
            "maximum_absolute_iteration_difference": max(
                abs(item["iteration_difference"]) for item in paired
            ),
            "lm_fewer_iterations": sum(item["iteration_difference"] < 0 for item in paired),
            "same_iterations": sum(item["iteration_difference"] == 0 for item in paired),
            "lm_more_iterations": sum(item["iteration_difference"] > 0 for item in paired),
        },
        "forecasts": forecasts,
        "records": records,
    }

    assert len(records) == len(FRACTIONS) * REPEATS * len(TRAIN_DATES) * len(backends)
    assert len(forecasts) == REPEATS * len(TEST_DATES) * len(backends)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
