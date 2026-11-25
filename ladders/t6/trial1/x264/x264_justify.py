"""Retrospective evidence and justification for the best observed X264 row."""
import csv
import json
import math
import sys

TARGET = "Performance-"

def _number(value):
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) else None
    except (AttributeError, TypeError, ValueError):
        return None

def justify(path):
    """Select the lowest finite outcome, retaining all setting cells."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if TARGET not in fields:
            raise ValueError("dataset must contain Performance-")
        settings = [name for name in fields if name != TARGET]
        first = None
        best = None
        usable = 0
        for record_id, row in enumerate(reader):
            outcome = _number(row.get(TARGET, ""))
            if record_id == 0:
                first = {"record_id": record_id, "outcome": outcome,
                         "configuration": {name: row.get(name, "") for name in settings}}
            if outcome is None:
                continue
            usable += 1
            candidate = {"record_id": record_id, "outcome": outcome,
                         "configuration": {name: row.get(name, "") for name in settings}}
            if best is None or outcome < best["outcome"]:
                best = candidate
        if best is None:
            raise ValueError("no usable finite Performance- records")
        comparison = {"status": "unavailable", "absolute_difference": None,
                      "percentage_improvement": None}
        if first is not None and first["outcome"] is not None:
            difference = first["outcome"] - best["outcome"]
            comparison = {"status": "computed", "absolute_difference": difference,
                          "percentage_improvement": (difference / first["outcome"] * 100
                                                      if first["outcome"] != 0 else None)}
            if first["outcome"] == 0:
                comparison["status"] = "zero_reference"
        return {"target": TARGET, "direction": "lower Performance- is better",
                "record_identifier": "zero-based CSV data-row index",
                "recommendation": best, "historical_reference": first,
                "comparison": comparison, "usable_record_count": usable,
                "evidence_label": "retrospective measured dataset evidence",
                "limitations": ["Associations do not establish the effect of changing a setting.",
                                "Correlated settings can share explanatory information.",
                                "This recommendation does not guarantee future performance or statistical superiority."],
                "assumptions": ["The CSV header defines the complete configuration.",
                                "Blank, '?', nonnumeric, and nonfinite outcomes are unusable.",
                                "Ties retain the earliest CSV record."]}

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("usage: x264_justify.py DATASET", file=sys.stderr)
        return 2
    print(json.dumps(justify(argv[0]), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
