"""Retrospective, observed-data justification for one configuration."""
import csv
import math
from pathlib import Path

CONFIGURATION_FIELDS = ["N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth"]
OUTCOME_FIELDS = ["MRE-", "ACC+", "PRED40+"]
DEMONSTRATION_POLICY = "lowest observed MRE-, ties by higher ACC+, then higher PRED40+, then earliest CSV record"


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _configuration(row):
    return {field: row.get(field) for field in CONFIGURATION_FIELDS}


def _median_reference(rows, field):
    values = sorted(value for value in (_finite(row.get(field)) for row in rows)
                    if value is not None)
    if not values:
        return {"name": "dataset finite-value median", "valid_count": 0, "value": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"name": "dataset finite-value median", "valid_count": len(values), "value": median}


def justify(path):
    """Return measured recommendation evidence; no prediction or causal claim is made."""
    rows = _rows(path)
    eligible = []
    for index, row in enumerate(rows):
        values = {field: _finite(row.get(field)) for field in OUTCOME_FIELDS}
        if all(value is not None for value in values.values()):
            eligible.append((index, row, values))
    result = {"policy": DEMONSTRATION_POLICY, "record_count": len(rows),
              "eligible_count": len(eligible), "references": {},
              "recommendation": None, "historical_references": {}}
    for field in OUTCOME_FIELDS:
        result["references"][field] = _median_reference(rows, field)
    if not eligible:
        result["status"] = "no eligible records: all three outcomes must be finite"
        return result
    index, row, values = min(eligible, key=lambda item: (item[2]["MRE-"],
                                                          -item[2]["ACC+"],
                                                          -item[2]["PRED40+"], item[0]))
    measured = {field: values[field] for field in OUTCOME_FIELDS}
    comparisons = {}
    for field, reference in result["references"].items():
        comparisons[field] = {"reference_name": reference["name"],
                              "reference_valid_count": reference["valid_count"],
                              "observed_minus_reference": (measured[field] - reference["value"]
                                                            if reference["value"] is not None else None)}
    result["status"] = "eligible recommendation found"
    result["recommendation"] = {"original_record_id": index, "csv_record_number": index + 1,
                                 "configuration": _configuration(row),
                                 "measured_outcomes": measured,
                                 "comparisons_with_historical_references": comparisons}
    result["interpretation"] = ("The recommendation is retrospective observed-data evidence. The best MRE- "
                                 "need not be best ACC+ or PRED40+; settings can be correlated and share "
                                 "explanatory information. Differences are not additive local contributions "
                                 "or causal effects, and future performance is not guaranteed.")
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    print(json.dumps(justify(parser.parse_args().dataset), indent=2, sort_keys=True))
