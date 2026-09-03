"""Tests for the Python adapter around the native LM matrix entry point."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import numpy as np
import pytest

from lm_idnet.algorithms import lm_native
from lm_idnet.algorithms.lm_native import NativeLmKernel
from lm_idnet.algorithms.log_likelihood import ScipyLogLikelihood
from lm_idnet.exceptions import (
    CommandUnavailableError,
    DataValidationError,
    NumericalPrecisionError,
)

pytestmark = [pytest.mark.integration, pytest.mark.numerical]


def test_native_adapter_evaluates_rows_independently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    alpha = np.array([1.0, 1.0])
    counts = np.array([[2, 1], [0, 1]]).T
    kernel = NativeLmKernel(precision_digits=6)

    native_result = kernel.calculate(counts, alpha)
    scipy_result = ScipyLogLikelihood().calculate(counts, alpha)
    aggregated_result = kernel.calculate(counts.sum(axis=0, keepdims=True), alpha)

    assert not counts.flags.c_contiguous
    assert native_result == pytest.approx(scipy_result, rel=1e-6, abs=1e-8)
    assert native_result != pytest.approx(aggregated_result, rel=1e-6)
    assert kernel.calculate(np.zeros((3, 2)), alpha) == 0.0


@pytest.mark.parametrize("precision_digits", [0, -1, 1.5, True])
def test_native_adapter_rejects_invalid_precision(
    precision_digits: object,
) -> None:
    with pytest.raises(DataValidationError, match="positive integer"):
        NativeLmKernel(precision_digits)  # type: ignore[arg-type]


def test_native_adapter_reports_library_and_coefficient_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_native_directory = lm_native._NATIVE_DIR
    monkeypatch.setattr(lm_native, "_NATIVE_DIR", tmp_path)
    with pytest.raises(CommandUnavailableError, match="unavailable or incompatible"):
        NativeLmKernel(6)

    shutil.copy2(original_native_directory / "LM_time_lib.so", tmp_path)
    with pytest.raises(CommandUnavailableError, match="status 1"):
        NativeLmKernel(6)


def test_native_adapter_translates_native_failure() -> None:
    kernel = NativeLmKernel(precision_digits=14)

    with pytest.raises(NumericalPrecisionError, match="14 significant digits"):
        kernel.calculate(np.array([[2, 0], [1, 1]]), np.ones(2))


def test_native_adapter_serializes_concurrent_calls() -> None:
    kernel = NativeLmKernel(precision_digits=6)
    cases = [
        (
            np.tile(np.array([[30, 0, 4, 1]]), (20, 1)),
            np.array([0.7, 2.3, 4.1, 1.2]),
        ),
        (
            np.tile(np.array([[1, 20, 3, 2]]), (20, 1)),
            np.array([3.0, 0.4, 1.5, 2.2]),
        ),
    ]
    expected = [ScipyLogLikelihood().calculate(*case) for case in cases]

    def calculate(index: int) -> tuple[int, float]:
        case_index = index % len(cases)
        return case_index, kernel.calculate(*cases[case_index])

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(calculate, range(200))

    for case_index, result in results:
        assert result == pytest.approx(expected[case_index], rel=1e-6)
