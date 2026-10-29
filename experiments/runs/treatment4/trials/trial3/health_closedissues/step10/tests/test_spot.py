import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from spot import screen


def test_robust_flags_and_counts():
    rows = [{"record_id": "a", "MRE-": 1, "ACC+": 10, "PRED40+": 5},
            {"record_id": "b", "MRE-": 2, "ACC+": 11, "PRED40+": 6},
            {"record_id": "c", "MRE-": 3, "ACC+": 12, "PRED40+": 7},
            {"record_id": "z", "MRE-": 100, "ACC+": 13, "PRED40+": 8}]
    result = screen(rows)
    assert result["records"][-1]["outcomes"]["MRE-"]["flagged"]
    assert result["flagged_record_count"] == 1
    assert result["flag_count_by_outcome"]["MRE-"] == 1
    assert [x["record_id"] for x in result["records"]] == ["a", "b", "c", "z"]


def test_missing_small_empty_constant_zero_scale_and_json():
    result = screen([{"record_id": "x", "MRE-": "nan", "ACC+": 1, "PRED40+": 1},
                     {"record_id": "y", "MRE-": "inf", "ACC+": 1, "PRED40+": 2}])
    assert result["records"][0]["outcomes"]["MRE-"]["score"] is None
    assert screen([])["flagged_record_count"] == 0
    constant = screen([{"MRE-": 2, "ACC+": 1, "PRED40+": 1},
                       {"MRE-": 2, "ACC+": 1, "PRED40+": 1},
                       {"MRE-": 3, "ACC+": 1, "PRED40+": 1}])
    assert constant["records"][-1]["outcomes"]["MRE-"]["score"] is None
    json.dumps(constant, allow_nan=False)
