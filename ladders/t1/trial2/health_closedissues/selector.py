#!/usr/bin/env python3
"""Compare a race-style configuration selector with a historical baseline."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

@dataclass(frozen=True)
class Record:
    values: dict
    utility: float = 0.0

def read_records(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(k not in rows[0] for k in OUTCOMES + CONFIG):
        raise ValueError("dataset is missing required columns")
    return [Record(r) for r in rows]

def utilities(records):
    bounds = {}
    for key in OUTCOMES:
        vals = [float(r.values[key]) for r in records]
        bounds[key] = (min(vals), max(vals))
    result = []
    for r in records:
        parts = []
        for key in OUTCOMES:
            lo, hi = bounds[key]
            value = float(r.values[key])
            scaled = 0.5 if hi == lo else (value - lo) / (hi - lo)
            if key == "MRE-":
                scaled = 1.0 - scaled
            parts.append(scaled)
        result.append(Record(r.values, sum(parts) / len(parts)))
    return result

def race_select(records, rounds=4):
    """Race candidates by utility, retaining progressively fewer survivors."""
    active = list(records)
    for _ in range(max(1, rounds)):
        active.sort(key=lambda r: (-r.utility, tuple(r.values[k] for k in CONFIG)))
        keep = max(1, (len(active) + 1) // 2)
        active = active[:keep]
    return max(active, key=lambda r: (r.utility, tuple(r.values[k] for k in CONFIG)))

def historical_baseline(records):
    """Historical practice: choose the row with the highest ACC, then earliest row."""
    return max(enumerate(records), key=lambda item: (float(item[1].values["ACC+"]), -item[0]))[1]

def compare(path):
    scored = utilities(read_records(path))
    selected = race_select(scored)
    baseline = historical_baseline(scored)
    return {
        "rows": len(scored), "selector_utility": selected.utility,
        "baseline_utility": baseline.utility,
        "selector": selected.values, "baseline": baseline.values,
        "winner": "selector" if selected.utility > baseline.utility else "baseline" if baseline.utility > selected.utility else "tie",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = compare(args.dataset)
    print(f"rows={result['rows']}")
    print(f"selector_utility={result['selector_utility']:.9f}")
    print(f"baseline_utility={result['baseline_utility']:.9f}")
    print(f"winner={result['winner']}")
    print("selector_config=" + repr({k: result["selector"][k] for k in CONFIG}))
    print("baseline_config=" + repr({k: result["baseline"][k] for k in CONFIG}))

if __name__ == "__main__":
    main()
