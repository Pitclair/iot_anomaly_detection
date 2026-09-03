"""Thread-safe ctypes bridge to the bundled Languasco-Migliardi kernel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import Lock

import numpy as np

from lm_idnet.exceptions import (
    CommandUnavailableError,
    DataValidationError,
    NumericalPrecisionError,
)

_NATIVE_DIR = Path(__file__).with_name("native")
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
_LM_LOCK = Lock()


class NativeLmKernel:
    """Evaluate matrices while serializing access to native global state."""

    def __init__(self, precision_digits: int) -> None:
        if (
            isinstance(precision_digits, bool)
            or not isinstance(precision_digits, int)
            or precision_digits <= 0
        ):
            raise DataValidationError(
                "LM precision_digits must be a positive integer"
            )

        library_path = _NATIVE_DIR / "LM_time_lib.so"
        try:
            library = ctypes.CDLL(str(library_path))
            library.initBern_logL_paths.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            library.initBern_logL_paths.restype = ctypes.c_int
            library.loggamma_LM_matrix.argtypes = [
                ctypes.c_int,
                _DOUBLE_POINTER,
                ctypes.c_long,
                ctypes.c_int,
                _DOUBLE_POINTER,
                ctypes.c_double,
            ]
            library.loggamma_LM_matrix.restype = ctypes.c_double
        except (AttributeError, OSError) as error:
            raise CommandUnavailableError(
                f"native LM library is unavailable or incompatible: {library_path}"
            ) from error

        with _LM_LOCK:
            status = library.initBern_logL_paths(
                os.fsencode(_NATIVE_DIR / "bernreal-norm-100.txt"),
                os.fsencode(_NATIVE_DIR / "err_coeff-100.txt"),
            )
        if status != 0:
            raise CommandUnavailableError(
                f"native LM coefficient initialization failed with status {status}"
            )

        self.precision_digits = precision_digits
        self._library = library

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        """Return the row-wise parameter-dependent likelihood-kernel sum."""
        contiguous_counts = np.ascontiguousarray(counts, dtype=np.float64)
        concentration = float(alpha.sum())
        probabilities = np.ascontiguousarray(
            alpha / concentration,
            dtype=np.float64,
        )

        with _LM_LOCK:
            result = self._library.loggamma_LM_matrix(
                self.precision_digits,
                contiguous_counts.ctypes.data_as(_DOUBLE_POINTER),
                contiguous_counts.shape[0],
                contiguous_counts.shape[1],
                probabilities.ctypes.data_as(_DOUBLE_POINTER),
                1.0 / concentration,
            )
        if not np.isfinite(result):
            raise NumericalPrecisionError(
                "native LM kernel could not satisfy "
                f"{self.precision_digits} significant digits"
            )
        return float(result)
