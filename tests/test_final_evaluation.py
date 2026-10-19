"""Protect the final-only evaluation boundary and exact annotation matching."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest

from lm_idnet.exceptions import DataValidationError
from lm_idnet.processing.schemas import ProcessedDataset
from lm_idnet.processing.storage import save_processed_dataset
from unsw_iot_attack_pcaps.prepare_labels import label_window

load_final = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_final.py"))["load_final"]
pytestmark = pytest.mark.unit


@pytest.fixture
def final_input(tmp_path, config_factory, window_factory):
    config = config_factory(ingest={"processed_root": str(tmp_path / "processed")})
    capture = config.ingest.partitions.final_test[0]
    window = window_factory()
    data = ProcessedDataset(metadata={"device_id": config.ingest.device_id,
        "capture_id": capture, "date": capture, "partition": "final_test", "file_source": "test.pcap"},
        windows=(window,))
    # Keep all non-final partitions absent: evaluating this boundary must not open them.
    config.ingest.partitions = config.ingest.partitions.model_copy(update={"final_test": (capture,)})
    data_path = config.ingest.processed_root / config.ingest.dataset_folder / f"{capture}.json"
    save_processed_dataset(data, data_path)
    label_path = tmp_path / "labels" / config.ingest.dataset_folder / f"{capture}.json"
    label_path.parent.mkdir(parents=True)
    digest = hashlib.sha256(b"test annotations").hexdigest()
    label = {"capture_id": capture, "device_id": config.ingest.device_id,
             "annotation_fingerprint": "sha256:" + digest,
             "windows": [label_window(window.start_utc, window.end_utc, ())[0]]}
    label_path.write_text(json.dumps(label))
    return config, data_path, label_path, digest


def test_final_loader_uses_only_final_captures(final_input):
    config, _, _, digest = final_input
    samples = load_final(config, (), digest)
    assert len(samples) == 1 and samples[0][0] == config.ingest.partitions.final_test[0]
    assert samples[0][2] is False


@pytest.mark.parametrize("fault", ["partition", "device", "duplicate", "provenance", "boolean", "duration"])
def test_final_loader_rejects_mismatched_inputs(final_input, fault):
    config, data_path, label_path, digest = final_input
    data, labels = json.loads(data_path.read_text()), json.loads(label_path.read_text())
    if fault == "partition":
        data["metadata"]["partition"] = "development_test"
    elif fault == "device":
        labels["device_id"] = "other-device"
    elif fault == "duplicate":
        labels["windows"].append(labels["windows"][0])
    elif fault == "provenance":
        labels["annotation_fingerprint"] = "sha256:wrong"
    elif fault == "boolean":
        labels["windows"][0]["is_attack"] = 0
    elif fault == "duration":
        config.ingest.window_minutes = 20
    data_path.write_text(json.dumps(data))
    label_path.write_text(json.dumps(labels))
    with pytest.raises((ValueError, DataValidationError)):
        load_final(config, (), digest)


def test_final_loader_checks_actual_time_across_captures(final_input):
    config, data_path, label_path, digest = final_input
    second = "2020-10-22"
    config.ingest.partitions = config.ingest.partitions.model_copy(
        update={"final_test": (config.ingest.partitions.final_test[0], second)})
    data, labels = json.loads(data_path.read_text()), json.loads(label_path.read_text())
    data["metadata"]["capture_id"] = labels["capture_id"] = second
    data["metadata"]["date"] = second
    data_path.with_name(second + ".json").write_text(json.dumps(data))
    label_path.with_name(second + ".json").write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="overlap"):
        load_final(config, (), digest)
