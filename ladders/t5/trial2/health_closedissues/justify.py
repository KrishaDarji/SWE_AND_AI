"""Retrospective configuration justification using standard-library CSV handling."""
import csv, math, statistics

FIELDS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = ["MRE-", "ACC+", "PRED40+"]

def finite(x):
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def read_records(path):
    records = []
    with open(path, newline="", encoding="utf-8") as stream:
        for record_id, row in enumerate(csv.DictReader(stream), 1):
            outcomes = {name: finite(row.get(name)) for name in OUTCOMES}
            settings = {name: row.get(name, "") for name in FIELDS}
            records.append({"record_id": record_id, "outcomes": outcomes, "settings": settings})
    return records

def eligible(record):
    return all(finite(record["outcomes"][name]) is not None for name in OUTCOMES)

def choose(records):
    candidates = [r for r in records if eligible(r)]
    if not candidates:
        return None, []
    chosen = min(candidates, key=lambda r: (r["outcomes"]["MRE-"], -r["outcomes"]["ACC+"], -r["outcomes"]["PRED40+"], r["record_id"]))
    return chosen, candidates

def medians(records):
    result = {}
    for name in OUTCOMES:
        values = [finite(r["outcomes"][name]) for r in records]
        values = [value for value in values if value is not None]
        result[name] = {"median": statistics.median(values) if values else None, "valid_count": len(values)}
    return result

def evidence(record, references):
    return {name: {"value": record["outcomes"][name], "reference_median": references[name]["median"], "valid_count": references[name]["valid_count"], "label": "measured"} for name in OUTCOMES}

def render(records):
    chosen, candidates = choose(records)
    refs = medians(records)
    lines = ["Configuration justification (retrospective observed-data evidence)", "Demonstration policy: lowest observed MRE-; ties by higher ACC+, then higher PRED40+, then earliest CSV record.", "Identifier: 1-based data-row position after the CSV header; preserved when records are excluded.", "Eligibility: all three outcomes must be finite numeric values."]
    for name in OUTCOMES:
        lines.append(f"Reference {name}: median={refs[name]['median']!r}, valid_count={refs[name]['valid_count']}")
    if chosen is None:
        lines.append("No eligible records; no recommendation is made.")
        return "\n".join(lines) + "\n"
    lines += [f"Eligible records: {len(candidates)} of {len(records)}", f"Recommended record identifier: {chosen['record_id']}", "Configuration: " + ", ".join(f"{k}={chosen['settings'].get(k, '')}" for k in FIELDS)]
    for name in OUTCOMES:
        value = chosen["outcomes"][name]
        median = refs[name]["median"]
        relation = "above" if value > median else "below" if value < median else "equal to"
        lines.append(f"Measured {name}: {value!r}; {relation} its finite-data median ({median!r}); valid_count={refs[name]['valid_count']}; evidence_label=measured")
    lines += ["Trade-off: the best observed MRE- need not also have the best ACC+ or PRED40+; the stated policy resolves this by tie-breaks only.", "Associations: settings and outcomes are jointly observed; correlated settings can share explanatory information. These global scores are not additive local contributions or causal effects.", "No model estimates are included. This is retrospective observed-data evidence, not a guarantee of future performance or a claim of statistical superiority.", "Limitations: one dataset and observed configurations; missing/nonfinite outcomes are excluded per-outcome from reference medians and from selection eligibility; no resource costs, units, ranges, or future constraints are inferred."]
    return "\n".join(lines) + "\n"

def main(path):
    print(render(read_records(path)), end="")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
