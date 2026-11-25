"""Budget-limited configuration selection for lower-is-better measurements."""
import math
import random


def _key(config):
    return tuple(sorted(config.items())) if isinstance(config, dict) else tuple(config)


def select(candidates, evaluate, budget, seed=42):
    """Evaluate at most budget distinct candidates and return an audit record."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")
    unique = []
    seen = set()
    for candidate in candidates:
        key = _key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    order = list(range(len(unique)))
    random.Random(seed).shuffle(order)
    limit = min(budget, len(order))
    observations = []
    evaluated = set()
    for index in order[:limit]:
        candidate = unique[index]
        key = _key(candidate)
        if key in evaluated:
            continue
        value = evaluate(candidate)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("evaluation result must be finite numeric")
        evaluated.add(key)
        observations.append({"evaluation": len(observations) + 1, "identifier": index + 1,
                             "value": value, "configuration": candidate})
    best = min(observations, key=lambda row: (row["value"], row["identifier"]))
    return {"evaluation_order": [row["identifier"] for row in observations],
            "observations": observations, "budget": budget,
            "budget_used": len(observations), "recommended_identifier": best["identifier"],
            "recommended_configuration": best["configuration"]}


def read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = []
    outcomes = []
    for row in rows:
        outcome = float(row.pop("Performance-"))
        candidates.append(row)
        outcomes.append(outcome)
    return candidates, outcomes


if __name__ == "__main__":
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _ in range(7):
        root = os.path.dirname(root)
    data = os.path.join(root,
                        "moot", "optimize", "config", "X264_AllMeasurements.csv")
    candidates, outcomes = read_csv(data)
    result = select(candidates, lambda candidate: outcomes[candidates.index(candidate)], 20, 42)
    print(json.dumps(result, sort_keys=True))
