"""Retrospective recommendation of an observed complete configuration."""
import csv
import math
import statistics
import sys
from summarize_runs import CONFIG, OUTCOMES, MISSING, finite_number
from blame import rank_associations

def _setting(row, name):
    value = row.get(name, "")
    if name == "criterion":
        value = value.strip()
        return None if value.lower() in MISSING else value
    return finite_number(value)

def _median(values):
    return statistics.median(values) if values else None

def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try: header = [x.strip() for x in next(reader)]
        except StopIteration: return header if 'header' in locals() else [], []
        result = []
        for position, raw in enumerate(reader, 2):
            if len(raw) == len(header):
                result.append((position, {header[i]: raw[i].strip() for i in range(len(header))}))
        return header, result

def justify(path):
    header, rows = _rows(path)
    eligible = []
    for position, row in rows:
        config = {name: _setting(row, name) for name in CONFIG}
        outcomes = {name: finite_number(row.get(name, "")) for name in OUTCOMES}
        if all(value is not None for value in config.values()) and all(value is not None for value in outcomes.values()):
            eligible.append((position, row, config, outcomes))
    references = {}
    for name in OUTCOMES:
        values = [finite_number(row.get(name, "")) for _, row in rows]
        values = [value for value in values if value is not None]
        references[name] = {"median": _median(values), "count": len(values), "direction": OUTCOMES[name]}
    result = {"status": "ok" if eligible else ("empty" if not rows else "no_eligible_recommendation"),
              "rows_seen": len(rows), "eligible_count": len(eligible), "references": references,
              "conventions": {"position": "header is CSV position 1; data rows retain original CSV positions starting at 2",
                               "malformed": "rows with the wrong field count are excluded; blank/missing/nonfinite settings or outcomes are ineligible",
                               "ties": "MRE- ascending, ACC+ descending, PRED40+ descending, then earliest original position"},
              "associations": None}
    if not eligible: return result
    selected = min(eligible, key=lambda item: (item[3]["MRE-"], -item[3]["ACC+"], -item[3]["PRED40+"], item[0]))
    position, row, config, outcomes = selected
    comparisons = {}
    for name, reference in references.items():
        median = reference["median"]
        difference = None if median is None else outcomes[name] - median
        comparisons[name] = {"selected": outcomes[name], "reference_median": median, "reference_count": reference["count"], "difference_selected_minus_median": difference, "direction": reference["direction"]}
    association = rank_associations(path)
    result.update({"recommendation": {"position": position, "configuration": config, "outcomes": outcomes},
                   "comparisons": comparisons,
                   "associations": {"label": "global marginal historical associations, not row-level effects", "outcome": association["outcome"], "grouping": "same categorical levels and numeric rank-balanced bins used by blame", "ranking": association["ranking"], "fields": association["fields"]},
                   "interpretation": "This is a retrospective recommendation from observed outcomes, not a prediction for an unevaluated configuration. Differences describe secondary-outcome tradeoffs; no units, ranges, runtime, resource cost, causality, or guaranteed benefit is inferred."})
    return result

def render(result):
    if result["status"] != "ok": return "Justification status: {}\nConventions: {}".format(result["status"], result["conventions"])
    r = result["recommendation"]
    lines = ["Retrospective configuration recommendation", "Status: ok", "Original position: {}".format(r["position"]), "Configuration: {}".format(r["configuration"]), "Observed outcomes: {}".format(r["outcomes"]), "Median references and selected-minus-median comparisons: {}".format(result["comparisons"]), "Selection rule: lowest MRE-, then highest ACC+, then highest PRED40+, then earliest original CSV position.", "Conventions: {}".format(result["conventions"]), result["interpretation"], "Association context: {}".format(result["associations"])]
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: justify.py DATASET.csv")
    print(render(justify(sys.argv[1])))
