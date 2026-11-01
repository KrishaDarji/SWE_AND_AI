"""Budgeted, callback-only selection of low Performance- X264 configurations."""
import csv
import json
import math
import random
import sys
from copy import deepcopy

TARGET = "Performance-"

def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None

def _key(configuration):
    return tuple(sorted((str(k), str(v)) for k, v in configuration.items()))

def choose(candidates, evaluate, budget=20, seed=42):
    """Evaluate at most budget unique candidates; evaluate is the only outcome source."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")
    unique = []
    seen = set()
    for identifier, configuration in enumerate(candidates):
        key = _key(configuration)
        if key not in seen:
            seen.add(key)
            unique.append((identifier, deepcopy(configuration)))
    order = list(unique)
    random.Random(seed).shuffle(order)
    observations = []
    for identifier, configuration in order[:min(budget, len(order))]:
        value = _finite(evaluate(deepcopy(configuration)))
        if value is None:
            raise ValueError("evaluation must return a finite number")
        observations.append({"record_id": identifier, "outcome": value,
                            "configuration": deepcopy(configuration)})
    if not observations:
        raise ValueError("no candidates were evaluated")
    recommendation = min(observations, key=lambda item: (item["outcome"], item["record_id"]))
    return {"target": TARGET, "direction": "lower Performance- is better",
            "seed": seed, "requested_budget": budget,
            "budget_used": len(observations), "unique_candidate_count": len(unique),
            "evaluation_order": [x["record_id"] for x in observations],
            "observations": deepcopy(observations),
            "recommended_identifier": recommendation["record_id"],
            "recommendation": deepcopy(recommendation),
            "record_identifier": "zero-based position in the supplied candidate list; duplicate settings retain the first position",
            "tie_breaking": "lowest outcome, then lowest record identifier",
            "limitations": ["Only evaluated candidates can be recommended.",
                            "Random exploration can miss a better unevaluated configuration.",
                            "No causal or future-performance guarantee is made."]}

def audit(candidates, outcomes, recommendation):
    """Retrospective full-data audit; it cannot mutate the frozen recommendation."""
    usable = [(i, _finite(outcomes[i])) for i in range(len(candidates)) if _finite(outcomes[i]) is not None]
    if not usable:
        return {"status": "unavailable", "observed_minimum": None, "gap": None}
    minimum = min(usable, key=lambda x: (x[1], x[0]))
    gap = recommendation["outcome"] - minimum[1]
    return {"status": "retrospective_only", "observed_minimum": {"record_id": minimum[0], "outcome": minimum[1]}, "gap": gap}

def _load(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0] or TARGET not in rows[0]:
        raise ValueError("dataset must contain rows and Performance-")
    settings = [name for name in rows[0] if name != TARGET]
    candidates = [{name: row.get(name, "") for name in settings} for row in rows]
    outcomes = [row.get(TARGET, "") for row in rows]
    return candidates, outcomes

def run(path, budget=20, seed=42):
    candidates, outcomes = _load(path)
    lookup = {_key(c): _finite(y) for c, y in zip(candidates, outcomes)}
    result = choose(candidates, lambda c: lookup[_key(c)], budget, seed)
    result["audit"] = audit(candidates, outcomes, result["recommendation"])
    return result

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("usage: x264_choose.py DATASET", file=sys.stderr)
        return 2
    print(json.dumps(run(argv[0]), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
