#!/usr/bin/env python3
"""Retrospective, observed-configuration justification for X264 measurements."""
import argparse
import csv
import math
import sys

def _number(value):
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None

def analyze(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        fields = rows.fieldnames or []
        if "Performance-" not in fields:
            raise ValueError("missing Performance- column")
        usable = []
        for record_number, row in enumerate(rows, 1):
            outcome = _number(row.get("Performance-"))
            if outcome is not None:
                usable.append((record_number, outcome, dict(row)))
    if not usable:
        return {"usable": 0, "message": "No usable records: Performance- had no finite numeric values."}
    reference_number, reference, reference_config = usable[0]
    best_number, best, best_config = min(usable, key=lambda item: (item[1], item[0]))
    difference = reference - best
    improvement = difference / reference * 100 if reference != 0 else None
    return {"usable": len(usable), "record_number": best_number, "outcome": best,
            "configuration": best_config, "reference_record_number": reference_number,
            "reference_outcome": reference, "reference_configuration": reference_config,
            "absolute_difference": difference, "percentage_improvement": improvement}

def render(result, path):
    if not result.get("usable"):
        return result["message"] + "\nAssumption: rows are CSV records after the header."
    config = result["configuration"]
    settings = ", ".join(f"{key}={value}" for key, value in config.items() if key != "Performance-")
    text = ["Measured result (observed dataset evidence):",
            f"- Recommended record identifier: CSV data record {result['record_number']} (1-based, excluding header).",
            f"- Measured Performance-: {result['outcome']:.3f}; lower is better.",
            f"- Complete configuration: {settings}.",
            f"Historical reference (first CSV data record, identifier {result['reference_record_number']}): Performance- {result['reference_outcome']:.3f}."]
    if result["percentage_improvement"] is None:
        text.append(f"- Absolute difference: {result['absolute_difference']:.3f}; percentage improvement is undefined because the reference outcome is zero.")
    else:
        text.append(f"- Absolute difference: {result['absolute_difference']:.3f}; percentage improvement: {result['percentage_improvement']:.3f}%.")
    text += ["This is a retrospective recommendation based on the lowest observed finite Performance-; it does not guarantee future performance or establish statistical superiority.",
             "Associations between settings and Performance- are descriptive only: they do not establish the effect of changing a setting, and correlated settings can share explanatory information.",
             "Assumptions and limitations: CSV order defines the tie-break and historical reference; missing/nonfinite outcomes are excluded; no causal, quality, resource, or constraint claims are made.",
             f"Input: {path}"]
    return "\n".join(text)

def main(argv=None):
    parser = argparse.ArgumentParser(description="justify an observed X264 configuration")
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    try:
        print(render(analyze(args.dataset), args.dataset))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
