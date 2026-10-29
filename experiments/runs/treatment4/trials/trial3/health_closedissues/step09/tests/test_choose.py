import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from choose import choose

FIELDS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth", "MRE-", "ACC+", "PRED40+"]


def write_rows(tmp_path, rows):
    path = tmp_path / "x.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return path


def row(n=50, criterion="squared_error", mre=1, acc=1, pred=1, complete=True):
    result = {"N_estimators": n, "criterion": criterion, "Min_sample_leaves": 1, "Min_impurity_decrease": 0, "Max_depth": 3,
              "MRE-": mre, "ACC+": acc, "PRED40+": pred}
    if not complete: result["criterion"] = ""
    return result


def test_best_and_cap_boundary(tmp_path):
    result = choose(write_rows(tmp_path, [row(100, mre=2), row(101, mre=0), row(100, mre=1)]), 100)
    assert result["original_record_id"] == 2 and result["feasible_count"] == 2


def test_all_tiebreakers(tmp_path):
    rows = [row(mre=1, acc=2, pred=1), row(mre=1, acc=2, pred=2), row(mre=1, acc=3, pred=0)]
    assert choose(write_rows(tmp_path, rows))["original_record_id"] == 2
    rows = [row(mre=1, acc=2, pred=2), row(mre=1, acc=2, pred=2)]
    assert choose(write_rows(tmp_path, rows))["original_record_id"] == 0


def test_invalid_caps_and_bad_values(tmp_path):
    path = write_rows(tmp_path, [row(n="nan"), row(complete=False)])
    for cap in (-1, math.inf, "100"):
        with pytest.raises(ValueError): choose(path, cap)
    result = choose(path); assert result["feasible_count"] == 0


def test_empty_and_no_feasible(tmp_path):
    path = write_rows(tmp_path, [])
    assert choose(path)["total_count"] == 0
    assert choose(write_rows(tmp_path, [row(101)]), 100)["status"] == "no feasible records"
