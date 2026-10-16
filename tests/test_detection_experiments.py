"""Check the experiment report's confusion matrix and tied ranking calculation."""

import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_experiment_metrics_handle_ties_and_normal_only_data():
    summarize = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_detection.py"))["summarize"]
    result = summarize([True, False, True, False], [True, True, False, False], [3., 2., 2., 1.])
    assert result["true_positive"] == result["false_positive"] == 1
    assert result["false_negative"] == result["true_negative"] == 1
    assert result["f1_score"] == .5
    assert result["roc_auc"] == .875
    normal = summarize([False, False], [True, False], [1., 0.])
    assert normal["f1_score"] is normal["roc_auc"] is None
    assert normal["false_positive_rate"] == .5
    assert summarize([], [True, False], [1., 0.]) == {"window_count": 2, "alert_count": 1}
    with pytest.raises(ValueError, match="every scored window"):
        summarize([True], [False, True], [1., 2.])
