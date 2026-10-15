"""Read bounded JSON lines and immediately emit the shared detector's events."""

from datetime import datetime, timedelta, timezone
from itertools import count
import json
import logging
from typing import BinaryIO, TextIO

from pydantic import BaseModel, ConfigDict, Field

from lm_idnet.artifacts import detector_state_lock
from lm_idnet.config import EdgeConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.processing.schemas import WindowRecord

logger = logging.getLogger(__name__)
MAX_INPUT_LINE_BYTES = 64 * 1024


class InputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    device_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    window: WindowRecord


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def run_windows(config: EdgeConfig, source: BinaryIO, destination: TextIO) -> None:
    """Process synchronously, retaining only the last boundary and bounded state."""
    previous_end = None
    interval = timedelta(minutes=config.ingest.window_minutes)
    origin = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with detector_state_lock(config.state_dir):
        detector = WindowDetector(config)
        for line_number in count(1):
            try:
                line = source.readline(MAX_INPUT_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_INPUT_LINE_BYTES:
                    raise DataValidationError(
                        f"record exceeds the {MAX_INPUT_LINE_BYTES}-byte line limit"
                    )
                record = InputRecord.model_validate(json.loads(
                    line.decode("utf-8"), parse_constant=_reject_nonfinite,
                ))
                window = record.window
                if previous_end is not None and window.start_utc < previous_end:
                    raise DataValidationError("window overlaps or precedes the previous window")
                if (window.start_utc - origin) % interval != timedelta(0):
                    raise DataValidationError("window start must align with the trained UTC boundaries")
                event, update, refit = detector.process(
                    window, device_id=record.device_id, capture_id=record.source_id,
                )
                previous_end = window.end_utc
                for attempt in (refit, update):
                    if attempt is not None:
                        logger.info("Detector update: %s", json.dumps(attempt, allow_nan=False))
                if event is not None:
                    destination.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
                    destination.flush()
            except DataValidationError as error:
                raise type(error)(f"input line {line_number}: {error}") from error
            except (ValueError, RecursionError, OSError) as error:
                raise DataValidationError(f"input line {line_number}: {error}") from error
        logger.info("Input closed after %d scored windows", detector.window_count)
