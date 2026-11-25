#!/usr/bin/env python3
"""Retrospective observed-data justification for a configuration."""
import csv, math, sys

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def median(values):
    values = sorted(values)
    if not values:
        return None
    n = len(values); mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2

def read_records(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [{"record_id": i, **row} for i, row in enumerate(csv.DictReader(handle), 1)]

def analyze(records):
    eligible = []
    for row in records:
        outcomes = {name: finite(row.get(name)) for name in OUTCOMES}
        if outcomes["MRE-"] is not None and outcomes["ACC+"] is not None and outcomes["PRED40+"] is not None:
            eligible.append((row, outcomes))
    if not eligible:
        return {"eligible_count": 0, "winner": None, "references": {name: {"median": None, "count": 0} for name in OUTCOMES}}
    winner, win_values = min(eligible, key=lambda item: (item[1]["MRE-"], -item[1]["ACC+"], -item[1]["PRED40+"], item[0]["record_id"]))
    references = {}
    for name in OUTCOMES:
        values = [values[name] for _, values in eligible]
        references[name] = {"median": median(values), "count": len(values)}
    return {"eligible_count": len(eligible), "winner": winner, "winner_values": win_values, "references": references}

def render(result):
    if result["winner"] is None:
        return "No eligible records: all records lacked at least one finite outcome.\n"
    row, values = result["winner"], result["winner_values"]
    lines = ["Retrospective observed-data justification", "Demonstration policy: lowest observed MRE-, ties by higher ACC+, then higher PRED40+, then earliest CSV record.", f"Eligible records: {result['eligible_count']}", f"Recommended original CSV record identifier: {row['record_id']}", "Configuration: " + ", ".join(f"{key}={row[key]}" for key in CONFIG), "Measured outcomes: " + ", ".join(f"{key}={values[key]}" for key in OUTCOMES), "Per-outcome references (finite eligible values):"]
    for name in OUTCOMES:
        ref = result["references"][name]
        relation = "below" if values[name] < ref["median"] else "above" if values[name] > ref["median"] else "equal to"
        lines.append(f"  {name}: median={ref['median']} (valid count={ref['count']}); recommendation is {relation} the reference")
    lines += ["Trade-off: the best MRE- need not also be the best ACC+ or PRED40+; these are reported separately, not added.", "Associations only: correlated settings can share explanatory information, so this record does not establish causal effects or independent feature contributions.", "Evidence label: measured results and medians are observed-data evidence; no model estimates are included.", "Limitations: missing/nonfinite outcomes are ineligible; original CSV identifiers are preserved; no future guarantee or statistical superiority is claimed.", "EZR-inspired machinery: CSV row streaming, finite numeric filtering, median summaries, stable ordering, and explicit best-row selection echo ezr-lib's csv/sorted/keysort patterns and ezr's ranked multi-goal selection. No Lua is executed."]
    return "\n".join(lines) + "\n"

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: justify.py DATASET.csv")
    result = analyze(read_records(argv[1]))
    print(render(result), end="")

if __name__ == "__main__":
    main(sys.argv)
