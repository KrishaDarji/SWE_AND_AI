import os
import tempfile

import pytest

from source.nasa93_race import classify_difference, compare, paired_difference

def test_paired_arithmetic_and_cases():
    assert paired_difference(2, 3) == -1
    assert classify_difference(-0.1) == "win"
    assert classify_difference(0.1) == "loss"
    assert classify_difference(1e-13) == "tie"

def test_comparison_is_reproducible_grouped_and_training_only():
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    handle.write("idX,a,b,Effort-,Kloc+,Defects-\n")
    for i, (a, b, y) in enumerate([("l", "x", 10), ("l", "x", 20), ("h", "y", 30), ("h", "y", 40), ("z", "q", 50), ("z", "q", 60)], 1):
        handle.write(f"{i},{a},{b},{y},1,1\n")
    handle.close()
    try:
        first = compare(handle.name, seeds=[42])
        second = compare(handle.name, seeds=[42])
        assert first == second
        item = first["results"][0]
        assert item["group_separation"]
        assert set(item["train_record_identifiers"]).isdisjoint(item["test_record_identifiers"])
        assert item["baseline_predictions"] == [item["training_mean"]] * item["test_count"]
        assert first["counts"]["win"] + first["counts"]["tie"] + first["counts"]["loss"] == 1
    finally:
        os.unlink(handle.name)

def test_held_out_outcome_change_does_not_change_predictions_or_baseline():
    def make(value, changed_id=None):
        h = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        h.write("idX,a,b,Effort-,Kloc+,Defects-\n1,l,x,10,1,1\n2,l,x,20,1,1\n3,h,y," + (str(value) if changed_id == 3 else "30") + ",1,1\n4,h,y," + (str(value) if changed_id == 4 else "40") + ",1,1\n")
        h.close()
        return h.name
    a = make(30)
    held_out_id = compare(a, seeds=[42])["results"][0]["test_record_identifiers"][0]
    b = make(999, held_out_id)
    try:
        left, right = compare(a, seeds=[42]), compare(b, seeds=[42])
        assert left["results"][0]["predictions"] == right["results"][0]["predictions"]
        assert left["results"][0]["baseline_predictions"] == right["results"][0]["baseline_predictions"]
    finally:
        os.unlink(a); os.unlink(b)
