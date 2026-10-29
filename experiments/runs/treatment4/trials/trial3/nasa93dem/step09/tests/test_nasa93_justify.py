import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from nasa93_justify import format_justification, justify
from nasa93_effort import evaluate
from nasa93_summary import read_csv

DATA = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


def test_justification_labels_and_proposal():
    report = justify(str(DATA))
    text = format_justification(report)
    assert report["descriptor_count"] == 22
    assert set(report["proposal"]) == set(evaluate(str(DATA))["predictor_names"])
    assert "MODEL ESTIMATE" in text and "AUDIT" in text
    assert "metadata and outcomes excluded" in text
    assert "marginal, not additive or causal" in text


def test_training_only_reference_and_outcome_independence(tmp_path):
    original = read_csv(str(DATA))
    changed = copy.deepcopy(original)
    heldout = evaluate(str(DATA))["test_rows"]
    selected_id = heldout[0]["idX"]
    for row in changed["rows"]:
        if row.get("idX") == selected_id:
            row["Effort-"] = 999999.0
    out = tmp_path / "changed.csv"
    with out.open("w") as handle:
        handle.write(",".join(changed["columns"]) + "\n")
        for row in changed["rows"]:
            handle.write(",".join("" if row.get(c) is None else str(row.get(c)) for c in changed["columns"]) + "\n")
    a, b = justify(str(DATA)), justify(str(out))
    assert a["estimate"] == pytest.approx(b["estimate"])
    assert format_justification(a).split("AUDIT")[0] == format_justification(b).split("AUDIT")[0]

