"""Paired race of the tuner and uniform random search for X264."""
import csv, json, math, random, sys
from copy import deepcopy
try:
    from .x264_choose import _finite, _key, choose, audit
except ImportError:
    from x264_choose import _finite, _key, choose, audit

TIE_TOLERANCE = 1e-12

def uniform_search(candidates, evaluate, budget=20, seed=42):
    """Baseline: seeded uniform sampling without replacement, then minimum."""
    # The tuner currently uses the same uniform acquisition; keeping this
    # separate makes equivalence explicit and leaves the callback boundary intact.
    result = choose(candidates, evaluate, budget, seed)
    result["strategy"] = "uniform_random_search_without_replacement"
    return result

def race(candidates, evaluate_tuner, evaluate_baseline, budget=20, seed=42,
         tie_tolerance=TIE_TOLERANCE):
    if not isinstance(tie_tolerance, (int, float)) or isinstance(tie_tolerance, bool) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be nonnegative and finite")
    tuner = choose(candidates, evaluate_tuner, budget, seed)
    baseline = uniform_search(candidates, evaluate_baseline, budget, seed)
    difference = tuner["recommendation"]["outcome"] - baseline["recommendation"]["outcome"]
    relation = "tie" if abs(difference) <= tie_tolerance else ("tuner_win" if difference < 0 else "tuner_loss")
    return {"seed": seed, "budget": budget, "tie_tolerance": tie_tolerance,
            "tuner": tuner, "baseline": baseline, "difference": difference,
            "relation": relation, "recommendations_frozen_before_audit": True,
            "interpretation": "difference = tuner Performance- minus baseline Performance-; lower is better"}

def load(path):
    with open(path, newline="", encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    if not rows or not rows[0] or "Performance-" not in rows[0]:
        raise ValueError("dataset must contain rows and Performance-")
    names = [x for x in rows[0] if x != "Performance-"]
    return ([{x: row.get(x, "") for x in names} for row in rows],
            [row.get("Performance-", "") for row in rows])

def run(path, budget=20, seeds=range(42, 52)):
    candidates, outcomes = load(path)
    lookup = {_key(c): _finite(y) for c, y in zip(candidates, outcomes)}
    reports = []
    for seed in seeds:
        # Each strategy gets a fresh counted callback and only sees callback outcomes.
        reports.append(race(candidates, lambda c, m=lookup: m[_key(c)],
                            lambda c, m=lookup: m[_key(c)], budget, seed))
        reports[-1]["tuner"]["audit"] = audit(candidates, outcomes, reports[-1]["tuner"]["recommendation"])
        reports[-1]["baseline"]["audit"] = audit(candidates, outcomes, reports[-1]["baseline"]["recommendation"])
    wins = sum(r["relation"] == "tuner_win" for r in reports); ties = sum(r["relation"] == "tie" for r in reports)
    losses = len(reports) - wins - ties
    return {"strategy_settings": {"budget": budget, "seeds": list(seeds), "without_replacement": True},
            "reports": reports, "summary": {"wins": wins, "ties": ties, "losses": losses,
            "mean_difference": sum(r["difference"] for r in reports) / len(reports) if reports else None,
            "equivalent": all(r["tuner"]["evaluation_order"] == r["baseline"]["evaluation_order"] for r in reports),
            "statistical_superiority": "pending"}}

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: x264_race.py DATASET")
    print(json.dumps(run(sys.argv[1]), indent=2, sort_keys=True))
