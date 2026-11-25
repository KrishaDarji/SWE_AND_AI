"""Compare the deterministic observed-data selector with a historical draw baseline."""
import csv
import math
import random
from pathlib import Path

from choose import CONFIGURATION_FIELDS, OUTCOME_FIELDS, _finite

TIE_TOLERANCE = 1e-12
SEEDS = tuple(range(42, 52))


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def eligible_records(path, cap=100):
    """Return the same complete, finite, cap-compliant records used by choose."""
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be a finite nonnegative number")
    records = []
    for index, row in enumerate(_rows(path)):
        config_ok = all(row.get(field) is not None and str(row.get(field)).strip()
                        for field in CONFIGURATION_FIELDS)
        config_ok = config_ok and all(_finite(row.get(field)) is not None
                                      for field in CONFIGURATION_FIELDS if field != "criterion")
        outcomes = {field: _finite(row.get(field)) for field in OUTCOME_FIELDS}
        n_estimators = _finite(row.get("N_estimators"))
        if config_ok and n_estimators is not None and n_estimators <= cap and all(
                value is not None for value in outcomes.values()):
            records.append({"original_record_id": index, "csv_record_number": index + 1,
                            "configuration": {field: row.get(field) for field in CONFIGURATION_FIELDS},
                            "measured_outcomes": outcomes})
    return records


def _selector(records):
    return min(records, key=lambda item: (item["measured_outcomes"]["MRE-"],
                                          -item["measured_outcomes"]["ACC+"],
                                          -item["measured_outcomes"]["PRED40+"],
                                          item["original_record_id"]))


def compare(path, cap=100, seeds=SEEDS, tie_tolerance=TIE_TOLERANCE):
    """Draw one eligible historical record per independent seed and pair it with selector."""
    if not isinstance(tie_tolerance, (int, float)) or isinstance(tie_tolerance, bool) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be a finite nonnegative number")
    records = eligible_records(path, cap)
    result = {"cap": cap, "tie_tolerance": tie_tolerance, "seeds": list(seeds),
              "eligible_count": len(records), "status": "no feasible records" if not records else "compared",
              "selector_strategy": "choose minimum observed MRE-, then higher ACC+, PRED40+, earliest record",
              "information_advantage": "selector inspects all feasible outcomes; baseline samples one record",
              "interpretation": "retrospective draws from one historical dataset, not independent validation samples or independent LLM trials",
              "runs": [], "summary": None}
    if not records:
        return result
    selected = _selector(records)
    wins = ties = losses = 0
    differences = []
    for seed in seeds:
        rng = random.Random(seed)
        baseline = rng.choice(records)
        difference = baseline["measured_outcomes"]["MRE-"] - selected["measured_outcomes"]["MRE-"]
        if difference > tie_tolerance:
            label = "selector win"; wins += 1
        elif difference < -tie_tolerance:
            label = "selector loss"; losses += 1
        else:
            label = "tie"; ties += 1
        differences.append(difference)
        result["runs"].append({"seed": seed, "selector": selected, "baseline": baseline,
                               "paired_mre_difference": difference, "classification": label,
                               "acc_plus_difference": baseline["measured_outcomes"]["ACC+"] - selected["measured_outcomes"]["ACC+"],
                               "pred40_plus_difference": baseline["measured_outcomes"]["PRED40+"] - selected["measured_outcomes"]["PRED40+"]})
    result["summary"] = {"count": len(differences), "mean": sum(differences) / len(differences),
                          "min": min(differences), "max": max(differences),
                          "wins": wins, "ties": ties, "losses": losses,
                          "tradeoff_note": "ACC+ and PRED40+ differences are descriptive paired trade-offs; no composite metric is formed."}
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--cap", type=float, default=100)
    args = parser.parse_args()
    print(json.dumps(compare(args.dataset, args.cap), indent=2, sort_keys=True))
