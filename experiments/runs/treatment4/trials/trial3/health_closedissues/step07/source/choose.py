"""Choose the best observed configuration under an explicit estimator cap."""
import csv
import math
from pathlib import Path

CONFIGURATION_FIELDS = ["N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth"]
OUTCOME_FIELDS = ["MRE-", "ACC+", "PRED40+"]


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _complete_configuration(row):
    if not all(row.get(field) is not None and str(row.get(field)).strip() for field in CONFIGURATION_FIELDS):
        return False
    return all(_finite(row.get(field)) is not None for field in CONFIGURATION_FIELDS if field != "criterion")


def choose(path, cap=100):
    """Select an eligible observed row; cap is an estimator-count compute proxy."""
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be a finite nonnegative number")
    rows = _read(path)
    eligible = []
    for index, row in enumerate(rows):
        estimator_count = _finite(row.get("N_estimators"))
        outcomes = {field: _finite(row.get(field)) for field in OUTCOME_FIELDS}
        if (_complete_configuration(row) and estimator_count is not None and estimator_count <= cap
                and all(value is not None for value in outcomes.values())):
            eligible.append((index, row, estimator_count, outcomes))
    result = {"status": "no feasible records" if not eligible else "feasible record selected",
              "cap": cap, "cap_definition": "N_estimators <= cap (compute proxy, not measured cost)",
              "feasible_count": len(eligible), "total_count": len(rows), "selection_basis": "observed outcomes"}
    if not eligible:
        result["selection_note"] = "Selection uses observed outcomes, not predictions for unseen settings."
        return result
    index, row, _, outcomes = min(eligible, key=lambda item: (item[3]["MRE-"], -item[3]["ACC+"],
                                                                -item[3]["PRED40+"], item[0]))
    result.update({"original_record_id": index, "configuration": {field: row.get(field) for field in CONFIGURATION_FIELDS},
                   "measured_outcomes": outcomes,
                   "selection_note": "Selection uses observed outcomes, not predictions for unseen settings.",
                   "limitation": "Other settings also affect compute; this estimator cap is only a proxy."})
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--cap", type=float, default=100)
    args = parser.parse_args()
    print(json.dumps(choose(args.dataset, args.cap), indent=2, sort_keys=True))
