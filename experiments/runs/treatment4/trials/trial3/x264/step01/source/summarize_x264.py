#!/usr/bin/env python3
"""Summarize an X264 measurements CSV without assuming domain-specific units."""
import argparse
import csv
import json
import math
from pathlib import Path

from sklearn.utils.validation import check_array


OUTCOME = "Performance-"


def _number(value):
    if value is None or not value.strip():
        return None
    try:
        number = float(value.strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def summarize(path):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    settings = [field for field in fields if field != OUTCOME]
    result = {
        "record_count": len(rows), "encoder_settings": settings,
        "outcome": {"name": OUTCOME, "direction": "lower is better"},
        "setting_frequencies": {}, "numeric_statistics": {},
        "missing_values": {}, "parsing_assumptions": [
            "The first CSV row is the header; comma separates fields.",
            "Blank or non-finite numeric cells are missing; other non-numeric cells are missing for numeric summaries.",
            "All non-outcome columns are reported as encoder settings; no units or formulas are inferred.",
        ],
        "limitations": [
            "Only the available Performance- outcome is summarized.",
            "Frequency counts are exact trimmed cell values and do not imply semantic categories.",
        ],
    }
    for field in fields:
        values = [row.get(field, "") for row in rows]
        result["missing_values"][field] = sum(_number(v) is None for v in values)
        if field != OUTCOME:
            counts = {}
            for value in values:
                key = value.strip()
                counts[key] = counts.get(key, 0) + 1
            result["setting_frequencies"][field] = dict(sorted(counts.items()))
    outcome_values = [_number(row.get(OUTCOME, "")) for row in rows]
    present = [v for v in outcome_values if v is not None]
    if present:
        # check_array supplies scikit-learn's finite numeric validation.
        checked = check_array(present, ensure_2d=False, dtype=float)
        mean = sum(checked) / len(checked)
        variance = sum((v - mean) ** 2 for v in checked) / len(checked)
        result["numeric_statistics"][OUTCOME] = {
            "count": len(checked), "min": min(checked), "max": max(checked),
            "mean": mean, "std_population": math.sqrt(variance),
            "median": sorted(checked)[len(checked) // 2] if len(checked) % 2 else
                (sorted(checked)[len(checked)//2-1] + sorted(checked)[len(checked)//2]) / 2,
        }
    else:
        result["numeric_statistics"][OUTCOME] = {"count": 0}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary = summarize(args.dataset)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
