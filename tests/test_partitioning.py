import pytest

from lm_idnet.config import AppConfig
from lm_idnet.partitioning import (
    all_capture_ids,
    calibration_partition_for_threshold,
    development_partition_for_evaluation,
    final_partition_for_locked_evaluation,
    fit_partition_for_training,
    partition_name_for_capture,
)

pytestmark = pytest.mark.unit


def test_partitions_are_disjoint_and_cover_whole_captures(
    config_factory,
) -> None:
    config: AppConfig = config_factory()
    selections = (
        fit_partition_for_training(config),
        calibration_partition_for_threshold(config),
        development_partition_for_evaluation(config),
        final_partition_for_locked_evaluation(config),
    )

    flattened = [
        capture_id
        for selection in selections
        for capture_id in selection.capture_ids
    ]

    assert len(flattened) == len(set(flattened))
    assert tuple(flattened) == all_capture_ids(config)


def test_training_cannot_receive_final_test_identifiers(config_factory) -> None:
    config: AppConfig = config_factory()
    final_ids = set(final_partition_for_locked_evaluation(config).capture_ids)
    training_ids = set(fit_partition_for_training(config).capture_ids)

    assert training_ids.isdisjoint(final_ids)


def test_threshold_calibration_cannot_receive_final_test_identifiers(
    config_factory,
) -> None:
    config: AppConfig = config_factory()
    final_ids = set(final_partition_for_locked_evaluation(config).capture_ids)
    calibration_ids = set(
        calibration_partition_for_threshold(config).capture_ids
    )

    assert calibration_ids.isdisjoint(final_ids)


def test_partition_accessors_return_immutable_capture_groups(
    config_factory,
) -> None:
    selection = fit_partition_for_training(config_factory())

    assert isinstance(selection.capture_ids, tuple)
    with pytest.raises(AttributeError):
        selection.capture_ids.append("camera-2099-01-01")


def test_every_capture_resolves_to_one_capture_level_partition(
    config_factory,
) -> None:
    config: AppConfig = config_factory()

    resolved = {
        capture_id: partition_name_for_capture(config, capture_id)
        for capture_id in all_capture_ids(config)
    }

    assert set(resolved.values()) == {
        "fit",
        "calibration",
        "development_test",
        "final_test",
    }
