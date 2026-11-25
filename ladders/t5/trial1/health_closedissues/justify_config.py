#!/usr/bin/env python3
"""Retrospective configuration justification for the Health CSV."""
import csv, math, statistics, sys
from pathlib import Path

FIELDS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = ["MRE-", "ACC+", "PRED40+"]

def finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False

def median_reference(rows, name):
    vals = [float(r[name]) for r in rows if finite(r.get(name))]
    return (statistics.median(vals), len(vals)) if vals else (None, 0)

def all_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        for record, row in enumerate(csv.DictReader(f), 1):
            row["record_id"] = str(record)
            yield row

def eligible_rows(path):
    return (r for r in all_rows(path) if all(finite(r.get(k)) for k in OUTCOMES))

def choose(rows):
    if not rows: return None
    return min(rows, key=lambda r: (float(r["MRE-"]), -float(r["ACC+"]),
                                    -float(r["PRED40+"]), int(r["record_id"])))

def fmt(x): return "unavailable" if x is None else f"{x:.12g}"

def analyze(path):
    rows = list(eligible_rows(path))
    original_rows = list(all_rows(path))
    refs = {k: median_reference(original_rows, k) for k in OUTCOMES}
    chosen = choose(rows)
    result = {"eligible_count": len(rows), "references": refs, "chosen": chosen}
    return result

def render(path):
    a = analyze(path); c = a["chosen"]
    lines = ["Retrospective observed-data configuration justification", "",
             "Demonstration policy: select the lowest observed MRE-, breaking ties by higher ACC+, then higher PRED40+, then earliest CSV record.",
             "Eligibility: a row is eligible only when all three outcomes are finite numeric values; its original CSV record number (header excluded) is preserved.",
             f"Eligible records: {a['eligible_count']}"]
    for k in OUTCOMES:
        v, n = a["references"][k]
        lines.append(f"Reference median {k}: {fmt(v)} (valid count: {n})")
    if c is None:
        lines += ["Recommendation: none; no eligible records were available.", "No configuration, outcomes, or comparisons are reported."]
        return "\n".join(lines) + "\n"
    lines += [f"Recommended configuration (observed record {c['record_id']}):"]
    for k in FIELDS: lines.append(f"  {k}: {c[k]}")
    lines += ["Measured outcomes:"]
    for k in OUTCOMES:
        v, n = a["references"][k]; delta = float(c[k]) - v if v is not None else None
        relation = "above" if delta is not None and delta > 0 else "below" if delta is not None and delta < 0 else "equal to" if delta is not None else "not comparable with"
        lines.append(f"  {k}: {c[k]} ({relation} reference median {fmt(v)})")
    lines += ["The MRE- choice is the demonstration policy result; it need not be best on ACC+ or PRED40+.",
              "These are measured results, not model estimates or causal effects. Settings can be correlated and may share explanatory information, so global scores are not additive local contributions.",
              "This is retrospective observed-data evidence, not a guarantee of future performance or statistical superiority.",
              "Assumptions and limitations: CSV order is the fallback identifier; only finite numeric outcomes are eligible; missing/nonfinite rows are excluded; no resource costs, metric ranges, formulas, or constraints are inferred.",
              "EZR-informed adaptations: csv-style streaming/coercion, stable deterministic ordering, and summary/reference arithmetic reflect ezr-lib.lua; the explicit goal direction and tie policy are specified for this demonstration. No Lua code is executed."]
    return "\n".join(lines) + "\n"

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: justify_config.py DATASET.csv")
    print(render(Path(argv[1])), end="")

if __name__ == "__main__": main(sys.argv)
