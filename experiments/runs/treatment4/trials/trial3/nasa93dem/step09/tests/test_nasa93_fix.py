import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from nasa93_effort import evaluate
from nasa93_fix import repair

DATA = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


def setup():
    result = evaluate(str(DATA))
    proposal = {name: result["test_rows"][0].get(name) for name in result["predictor_names"]}
    return result, proposal


def test_repair_is_training_only_and_reports_required_fields():
    result, proposal = setup()
    report = repair(result, proposal, 100)
    assert report["status"] in {"repair-suggested", "no-feasible-repair", "no-repair-needed"}
    assert set(report["original_descriptors"]) == set(result["predictor_names"])
    assert "budget" in report and "changed_fields" in report
    if report["status"] == "repair-suggested":
        assert report["candidate_record_identifier"] in [r.get("idX") for r in result["train_rows"]]
        assert report["candidate_estimated_effort"] <= 100


def test_already_feasible_returns_no_repair():
    result, proposal = setup()
    report = repair(result, proposal, 10**12)
    assert report["status"] == "no-repair-needed"
    assert report["change_count"] == 0


def test_priority_and_tie_breaking_with_stub_model():
    class Model:
        def predict(self, frame):
            return [float(len([v for v in row if v == "high"]) * 10 + len([v for v in row if v == "medium"]) * 5 + 1) for row in frame]

    result = {"predictor_names": ["a", "b"], "model": Model(), "train_rows": [
        {"idX": 20, "a": "medium", "b": "low"},
        {"idX": 10, "a": "low", "b": "low"},
        {"idX": 30, "a": "low", "b": "medium"},
    ]}
    report = repair(result, {"a": "high", "b": "low"}, 6)
    assert report["status"] == "repair-suggested"
    assert report["change_count"] == 1
    assert report["candidate_record_identifier"] == 10


def test_no_feasible_and_invalid_or_incomplete_inputs():
    result, proposal = setup()
    with pytest.raises(ValueError): repair(result, proposal, -1)
    with pytest.raises(ValueError): repair(result, {"missing": "x"}, 100)
    class Model:
        def predict(self, frame): return [999.0] * len(frame)
    impossible = {"predictor_names": ["a"], "model": Model(), "train_rows": [{"idX": 1, "a": "x"}]}
    assert repair(impossible, {"a": "y"}, 1)["status"] == "no-feasible-repair"


@pytest.mark.parametrize("prediction", [-3.0, float("nan"), float("inf")])
def test_invalid_candidate_predictions_are_excluded(prediction):
    class Model:
        def predict(self, frame):
            return [prediction if row[0] == "bad" else (20.0 if row[0] == "original" else 5.0) for row in frame]
    result = {"predictor_names": ["a"], "model": Model(), "train_rows": [
        {"idX": 1, "a": "bad"}, {"idX": 2, "a": "good"}]}
    report = repair(result, {"a": "original"}, 10)
    assert report["status"] == "repair-suggested"
    assert report["candidate_record_identifier"] == 2
    assert report["invalid_candidate_count"] == 1


@pytest.mark.parametrize("prediction", [-1.0, float("nan"), float("inf")])
def test_invalid_original_prediction_is_explicit(prediction):
    class Model:
        def predict(self, frame): return [prediction]
    result = {"predictor_names": ["a"], "model": Model(), "train_rows": [{"idX": 1, "a": "good"}]}
    report = repair(result, {"a": "original"}, 10)
    assert report["status"] == "invalid-original-estimate"
    assert report["candidate_descriptors"] is None


def test_no_valid_candidate_remaining():
    class Model:
        def predict(self, frame): return [5.0 if row[0] == "original" else -1.0 for row in frame]
    result = {"predictor_names": ["a"], "model": Model(), "train_rows": [{"idX": 1, "a": "bad"}]}
    report = repair(result, {"a": "original"}, 0)
    assert report["status"] == "no-feasible-repair"
    assert report["invalid_candidate_count"] == 1
