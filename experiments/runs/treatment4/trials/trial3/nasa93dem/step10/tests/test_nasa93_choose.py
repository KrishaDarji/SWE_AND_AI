import math
import pytest
from source.nasa93_choose import choose

COLUMNS = ["idX", "a", "b", "Kloc+", "Effort-", "Defects-", "MonthsX"]

def row(i, effort=10, size=20):
    return {"idX": i, "a": "h", "b": i, "Kloc+": size, "Effort-": effort, "Defects-": 1, "MonthsX": 3}

def test_boundaries_and_maximum_and_metadata_separation():
    result = choose([row(1, 100, 10), row(2, 100, 30), row(3, 101, 99)], COLUMNS, 100)
    assert result["record_identifier"] == 2
    assert result["feasible_count"] == 2 and result["total_count"] == 3
    assert "idX" not in result["descriptors"] and "MonthsX" not in result["descriptors"]
    assert set(result["descriptors"]) == {"a", "b"}

def test_ties_lower_effort_then_earliest_record():
    assert choose([row(1, 9, 20), row(2, 8, 20)], COLUMNS, 10)["record_identifier"] == 2
    assert choose([row(1, 8, 20), row(2, 8, 20)], COLUMNS, 10)["record_identifier"] == 1

@pytest.mark.parametrize("budget", [-1, float("inf"), "bad"])
def test_invalid_budget(budget):
    with pytest.raises(ValueError): choose([row(1)], COLUMNS, budget)

def test_missing_nonfinite_empty_and_no_feasible():
    with pytest.raises(ValueError): choose([], COLUMNS, 10)
    with pytest.raises(ValueError): choose([row(1, 10, None), row(2, math.nan, 20)], COLUMNS, 10)
    with pytest.raises(ValueError): choose([row(1, 11, 20)], COLUMNS, 10)
