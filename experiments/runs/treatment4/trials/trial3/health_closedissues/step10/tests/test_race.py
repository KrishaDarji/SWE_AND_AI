import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from race import compare, eligible_records

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def make_csv(tmp_path, rows):
    path = tmp_path / "race.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return path


def row(n=50, mre=1, acc=1, pred=1):
    return {"MRE-": mre, "ACC+": acc, "PRED40+": pred, "N_estimators": n,
            "criterion": "squared_error", "Min_sample_leaves": 1,
            "Min_impurity_decrease": 0, "Max_depth": 3}


def test_selector_reuse_cap_and_seed_reproducibility(tmp_path):
    path = make_csv(tmp_path, [row(50, 0.1, .2, .3), row(100, .2, .9, .9), row(101, 0)])
    a, b = compare(path), compare(path)
    assert a == b and a["eligible_count"] == 2
    assert all(run["selector"]["original_record_id"] == 0 for run in a["runs"])
    assert all(run["baseline"]["configuration"] == run["baseline"]["configuration"] for run in a["runs"])


def test_independent_rng_and_paired_arithmetic_with_win_loss_tie(tmp_path):
    path = make_csv(tmp_path, [row(mre=1, acc=1, pred=1), row(mre=2, acc=2, pred=2)])
    result = compare(path, seeds=[1, 2, 3], tie_tolerance=0)
    assert result["summary"]["wins"] + result["summary"]["ties"] + result["summary"]["losses"] == 3
    for run in result["runs"]:
        assert run["paired_mre_difference"] == run["baseline"]["measured_outcomes"]["MRE-"] - run["selector"]["measured_outcomes"]["MRE-"]
    assert compare(path, seeds=[0], tie_tolerance=10)["runs"][0]["classification"] == "tie"


def test_invalid_caps_and_no_eligible_records(tmp_path):
    path = make_csv(tmp_path, [row(101)])
    for cap in (-1, math.inf, "100"):
        with pytest.raises(ValueError): eligible_records(path, cap)
    assert compare(path)["status"] == "no feasible records"
