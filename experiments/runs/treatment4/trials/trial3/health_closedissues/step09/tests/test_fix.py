import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from fix import repair

FIELDS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth", "MRE-", "ACC+", "PRED40+"]


def write(tmp_path, rows):
    path = tmp_path / "x.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return path


def row(n=10, criterion="squared_error", leaves=2, impurity="0.1", depth=5, mre=10, acc=5, pred=4):
    return {"N_estimators": str(n), "criterion": criterion, "Min_sample_leaves": str(leaves),
            "Min_impurity_decrease": str(impurity), "Max_depth": str(depth), "MRE-": str(mre),
            "ACC+": str(acc), "PRED40+": str(pred)}


def test_minimum_change_beats_larger_improvement_and_reports_numeric_changes(tmp_path):
    p = write(tmp_path, [row(), row(criterion="absolute_error", mre=9, acc=1, pred=1), row(n=11, criterion="absolute_error", mre=1, acc=99, pred=99)])
    result = repair(p)
    assert result["candidate"]["original_record_id"] == 1
    assert result["candidate"]["change_count"] == 1
    assert result["candidate"]["changes"] == [{"field": "criterion", "old": "squared_error", "new": "absolute_error"}]
    assert result["candidate"]["mre_improvement"] == 1.0


def test_numeric_equivalence_and_all_tiebreakers(tmp_path):
    base = row(mre=10, acc=5, pred=4)
    rows = [base, row(n=11, mre=9, acc=5, pred=4), row(n=12, mre=9, acc=6, pred=1),
            row(n=13, mre=9, acc=6, pred=2), row(n=14, mre=9, acc=6, pred=2)]
    result = repair(write(tmp_path, rows))
    assert result["candidate"]["original_record_id"] == 3
    assert result["candidate"]["change_count"] == 1


def test_secondary_tradeoff_missing_baseline_and_invalid_no_improvement(tmp_path):
    p = write(tmp_path, [row(acc="", mre=10), row(n=11, mre=9, acc=1), row(n=12, mre=11)])
    result = repair(p)
    assert result["candidate"]["secondary_comparisons"]["ACC+"]["difference"] is None
    assert result["candidate"]["secondary_comparisons"]["ACC+"]["worsening"] is None
    assert "unknown" in result["candidate"]["secondary_comparisons"]["ACC+"]["reason"]
    assert repair(write(tmp_path, [row(mre=10), row(n=11, mre=10)]))["status"] == "no improving candidate"
    assert repair(write(tmp_path, [row(mre="nan")]))["status"].startswith("invalid baseline")
    assert repair(write(tmp_path, []))["status"] == "empty input"


if __name__ == "__main__":
    pytest.main([__file__])
