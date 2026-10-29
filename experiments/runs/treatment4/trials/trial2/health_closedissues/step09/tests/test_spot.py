import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from spot import screen_records


def test_unusual_and_ordinary_counts_and_arithmetic():
    rows = [{"MRE-": str(x), "ACC+": str(x), "PRED40+": str(x)} for x in [1, 2, 3, 4, 100]]
    result = screen_records(rows, threshold=3.5)
    mre = result["outcomes"][0]
    assert mre["median"] == 3 and mre["mad"] == 1 and mre["robust_scale"] == pytest.approx(1.4826)
    assert mre["flags"][0]["record_id"] == 5
    assert result["unique_flagged_record_count"] == 1
    assert result["per_outcome_flag_counts"] == {"MRE-": 1, "ACC+": 1, "PRED40+": 1}
    assert mre["flags"][0]["score"] == pytest.approx((100 - 3) / 1.4826)


def test_missing_nonfinite_empty_and_small_inputs():
    result = screen_records([{"MRE-": "", "ACC+": "nan", "PRED40+": "inf"}])
    assert all(item["status"] == "no_finite_observations" for item in result["outcomes"])
    empty = screen_records([])
    assert empty["unique_flagged_record_count"] == 0
    small = screen_records([{"MRE-": "2", "ACC+": "2", "PRED40+": "2"}])
    assert small["outcomes"][0]["flag_count"] == 0


def test_constant_and_zero_scale_are_explicit_and_json_safe():
    result = screen_records([{"MRE-": "1"}, {"MRE-": "1"}, {"MRE-": "2"}])
    item = result["outcomes"][0]
    assert item["robust_scale"] == 0 and item["flags"][0]["score"] is None
    assert json.dumps(result, allow_nan=False)
    constant = screen_records([{"MRE-": "1"}, {"MRE-": "1"}])
    assert constant["outcomes"][0]["flag_count"] == 0


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        screen_records([], math.inf)
