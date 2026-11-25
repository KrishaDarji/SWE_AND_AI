#!/usr/bin/env python3
"""Retrospective recommendation for an X264 measurement table."""
import csv
import math
import sys


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def analyze(rows):
    if not rows:
        return {"status": "no usable records", "reason": "empty input"}
    fields = list(rows[0])
    outcome = "Performance-"
    usable = []
    for index, row in enumerate(rows):
        value = finite_number(row.get(outcome))
        if value is not None:
            usable.append((value, index, row))
    if not usable:
        return {"status": "no usable records", "reason": "no finite Performance- values"}
    value, index, row = min(usable, key=lambda item: (item[0], item[1]))
    reference = finite_number(rows[0].get(outcome))
    result = {"status": "ok", "record_index": index + 1, "record_id": index + 1,
              "configuration": {key: row[key] for key in fields if key != outcome},
              "measured": value, "reference_measured": reference,
              "reference_record_id": 1, "outcome": outcome}
    if reference is None:
        result["comparison"] = "absolute difference and percentage improvement unavailable: reference is missing or invalid"
        result["absolute_difference"] = None
        result["percentage_improvement"] = None
    elif reference == 0:
        result["absolute_difference"] = reference - value
        result["percentage_improvement"] = None
        result["comparison"] = "absolute difference available; percentage improvement unavailable: reference is zero"
    else:
        result["absolute_difference"] = reference - value
        result["percentage_improvement"] = (reference - value) / reference * 100.0
        result["comparison"] = "available"
    return result


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def render(result):
    if result["status"] != "ok":
        return "No recommendation: %s (%s)." % (result["status"], result["reason"])
    lines = ["Retrospective dataset evidence (measured results only).",
             "Recommended record: %s (CSV data record %s)." % (result["record_id"], result["record_id"]),
             "Measured Performance-: %.12g (lower is better)." % result["measured"],
             "Complete configuration: %s" % ", ".join("%s=%s" % item for item in result["configuration"].items()),
             "Historical reference: first CSV data record (record 1), measured Performance-=%s." % result["reference_measured"]]
    if result["absolute_difference"] is not None:
        lines.append("Absolute difference (reference - recommendation): %.12g." % result["absolute_difference"])
    if result["percentage_improvement"] is not None:
        lines.append("Percentage improvement: %.12g%%." % result["percentage_improvement"])
    if result["percentage_improvement"] is None:
        lines.append(result["comparison"] + ".")
    lines += ["This is an observed recommendation, not a guarantee of future performance or a claim of statistical superiority.",
              "Associations, if computed, would not establish the effect of changing a setting; correlated settings can share explanatory information.",
              "Limitations: one retrospective CSV, no causal intervention, and invalid/nonfinite outcomes excluded."]
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: justify.py DATASET.csv")
    print(render(analyze(load(argv[1]))))


if __name__ == "__main__":
    main(sys.argv)
