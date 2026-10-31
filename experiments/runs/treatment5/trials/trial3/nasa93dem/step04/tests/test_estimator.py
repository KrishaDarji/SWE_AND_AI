import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow_estimator import DESCRIPTORS, CategoricalRegressor, assess

def row(i, effort, **changes):
    x = {name: "a" for name in DESCRIPTORS}
    x.update(changes)
    x.update({"idX": str(i), "Effort-": effort})
    return x

def test_warmup_and_arithmetic():
    result = assess([row(1, 10), row(2, 20), row(3, 30)])
    assert len(result) == 1 and result[0]["prior_training_count"] == 2
    assert result[0]["prediction"] == 15 and result[0]["baseline"] == 15
    assert result[0]["absolute_error"] == 15 and result[0]["cumulative_mae"] == 15

def test_empty_short_and_invalid_effort():
    assert assess([]) == []
    assert assess([row(1, 1)]) == []
    assert assess([row(1, "nan"), row(2, "inf"), row(3, 3)]) == []

def test_missing_unseen_and_exclusions():
    result = assess([row(1, 10), row(2, 20), row(3, 30, prec="unseen", rely="?")])
    assert result[0]["prediction"] == 15
    model = CategoricalRegressor().fit([({name: "a" for name in DESCRIPTORS}, 10),
                                        ({name: "b" for name in DESCRIPTORS}, 20)])
    assert len(model.descriptors) == 22 and "Kloc+" not in model.descriptors
    assert math.isfinite(model.predict({name: "never" for name in DESCRIPTORS}))

def test_current_and_future_outcomes_do_not_change_prediction():
    base = [row(1, 10), row(2, 20), row(3, 30), row(4, 40)]
    changed = [dict(item) for item in base]
    changed[2]["Effort-"] = 999
    changed[3]["Effort-"] = 888
    assert assess(base)[0]["prediction"] == assess(changed)[0]["prediction"]
    assert assess(base)[0]["cumulative_mae"] == 15

def test_prediction_is_before_reveal():
    records = [row(1, 10), row(2, 20), row(3, 30)]
    before = assess(records)[0]["prediction"]
    records[2]["Effort-"] = 3000
    assert before == assess(records)[0]["prediction"]
