"""Tests for canonical processed-timeline validation."""

from datetime import datetime, timedelta, timezone

import pytest

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.partitioning import partition_name_for_capture
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.storage import save_processed_dataset
from lm_idnet.processing.timeline import (
    build_canonical_timeline,
    load_canonical_partition,
    validate_unique_timeline,
)

pytestmark = pytest.mark.unit


def _config(config_factory, tmp_path, *, resolve: bool) -> AppConfig:
    capture_ids = tuple(f"camera-2020-01-0{day}" for day in range(1, 6))
    merges = (
        [
            {
                "capture_ids": capture_ids[:2],
                "target_capture_id": capture_ids[0],
                "window_start_utc": "2020-01-01T00:00:00Z",
                "window_end_utc": "2020-01-01T00:10:00Z",
                "reason": "Known complementary test fragments",
            }
        ]
        if resolve
        else []
    )
    return config_factory(
        ingest={
            "processed_root": tmp_path,
            "partitions": {
                "fit": capture_ids[:2],
                "calibration": [capture_ids[2]],
                "development_test": [capture_ids[3]],
                "final_test": [capture_ids[4]],
            },
            "window_fragment_merges": merges,
        },
        outputs={"reports_dir": tmp_path / "reports"},
    )


def _write_datasets(config: AppConfig) -> None:
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    for index, capture_id in enumerate(config.ingest.partitions.all_capture_ids()):
        start = datetime(2020, 1, max(1, index), tzinfo=timezone.utc)
        counts = (index + 1, 0, 0, 0)
        save_processed_dataset(
            ProcessedDataset(
                metadata=Metadata(
                    device_id=config.ingest.device_id,
                    capture_id=capture_id,
                    partition=partition_name_for_capture(config, capture_id),
                    date=capture_id,
                    file_source=f"{capture_id}.pcap",
                ),
                windows=(
                    WindowRecord(
                        start_utc=start,
                        end_utc=start + timedelta(minutes=10),
                        categories=config.ingest.categories,
                        counts=counts,
                        state="observed",
                    ),
                ),
            ),
            processed_dir / f"{capture_id}.json",
        )


def test_documented_fragments_are_merged_into_one_unique_window(
    tmp_path,
    config_factory,
) -> None:
    config = _config(config_factory, tmp_path, resolve=True)
    _write_datasets(config)
    report_path = tmp_path / "timeline.json"

    datasets, report = build_canonical_timeline(config)
    validated = validate_unique_timeline(config, report_path)

    assert report["accepted"] is True
    assert validated["accepted"] is True
    assert validated["resolved_window_count"] == 1
    assert validated["overlap_count"] == 0
    assert datasets[config.ingest.partitions.fit[0]].windows[0].counts == (3, 0, 0, 0)
    assert datasets[config.ingest.partitions.fit[1]].windows == ()
    assert report_path.is_file()


def test_unresolved_duplicate_window_is_rejected(
    tmp_path,
    config_factory,
) -> None:
    config = _config(config_factory, tmp_path, resolve=False)
    _write_datasets(config)

    with pytest.raises(DataValidationError, match="1 unresolved overlap"):
        load_canonical_partition(config, config.ingest.partitions.fit)
