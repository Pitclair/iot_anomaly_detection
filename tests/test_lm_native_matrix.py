"""Direct contract test for the native LM matrix entry point."""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from lm_idnet.algorithms.log_likelihood import ScipyLogLikelihood

pytestmark = [pytest.mark.integration, pytest.mark.numerical]


def test_native_matrix_entry_point_evaluates_rows_independently() -> None:
    native_directory = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lm_idnet"
        / "algorithms"
        / "native"
    )
    library = ctypes.CDLL(str(native_directory / "LM_time_lib.so"))
    double_pointer = ctypes.POINTER(ctypes.c_double)
    library.initBern_logL_paths.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.initBern_logL_paths.restype = ctypes.c_int
    library.loggamma_LM_matrix.argtypes = [
        ctypes.c_int,
        double_pointer,
        ctypes.c_long,
        ctypes.c_int,
        double_pointer,
        ctypes.c_double,
    ]
    library.loggamma_LM_matrix.restype = ctypes.c_double

    assert (
        library.initBern_logL_paths(
            str(native_directory / "bernreal-norm-100.txt").encode(),
            str(native_directory / "err_coeff-100.txt").encode(),
        )
        == 0
    )

    alpha = np.array([1.0, 1.0])
    probabilities = np.ascontiguousarray(alpha / alpha.sum())

    def calculate(counts: np.ndarray, precision_digits: int = 6) -> float:
        counts = np.ascontiguousarray(counts, dtype=np.float64)
        return float(
            library.loggamma_LM_matrix(
                precision_digits,
                counts.ctypes.data_as(double_pointer),
                counts.shape[0],
                counts.shape[1],
                probabilities.ctypes.data_as(double_pointer),
                1.0 / alpha.sum(),
            )
        )

    counts = np.array([[2, 0], [1, 1]])
    native_result = calculate(counts)
    scipy_result = ScipyLogLikelihood().calculate(counts, alpha)
    aggregated_result = calculate(counts.sum(axis=0, keepdims=True))

    assert native_result == pytest.approx(scipy_result, rel=1e-6, abs=1e-8)
    assert native_result != pytest.approx(aggregated_result, rel=1e-6)
    assert calculate(np.zeros((3, 2))) == 0.0
    assert np.isnan(calculate(np.array([[1.5, 0.0]])))
    assert np.isnan(calculate(counts, precision_digits=14))
