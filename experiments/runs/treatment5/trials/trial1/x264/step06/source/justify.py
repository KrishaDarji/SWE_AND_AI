#!/usr/bin/env python3
"""Retrospective configuration justification for X264 measurements."""
import csv
import math
import sys


def finite_number(value):
    try:
        number = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def analyze(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or "Performance-" not in fields:
            return {"status": "no_usable_records", "reason": "missing Performance- column"}
        configurations = [name for name in fields if name != "Performance-"]
        records = []
        for csv_record, row in enumerate(reader, start=2):
            outcome = finite_number(row.get("Performance-"))
            if outcome is not None:
                records.append({"csv_record": csv_record, "configuration":
                                {name: row.get(name, "") for name in configurations},
                                "performance": outcome})
        if not records:
            return {"status": "no_usable_records", "reason": "no finite Performance- values"}
        recommendation = min(records, key=lambda item: (item["performance"], item["csv_record"]))
        reference = records[0] if finite_number(str(records[0]["performance"])) is not None else None
        # The reference is explicitly the first CSV data record, even if unusable.
        with open(path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        raw_reference = rows[0].get("Performance-") if rows else None
        reference_value = finite_number(raw_reference)
        result = {"status": "ok", "fields": fields, "recommendation": recommendation,
                  "reference_raw": raw_reference, "reference_value": reference_value,
                  "record_count": len(records), "raw_record_count": len(rows)}
        if reference_value is not None:
            difference = reference_value - recommendation["performance"]
            result["absolute_difference"] = abs(difference)
            result["improvement_percent"] = (difference / reference_value * 100
                                              if reference_value != 0 else None)
        else:
            result["absolute_difference"] = None
            result["improvement_percent"] = None
        return result


def render(result):
    if result["status"] != "ok":
        return "Recommendation status: no usable records\nReason: " + result["reason"]
    chosen = result["recommendation"]
    lines = ["Recommendation status: retrospective dataset evidence",
             "Measured result (observed): Performance- = %.6f" % chosen["performance"],
             "Record identifier: CSV data record %d" % chosen["csv_record"],
             "Complete configuration: " + ", ".join("%s=%s" % pair
                                                     for pair in chosen["configuration"].items()),
             "Historical reference: first CSV data record (record 2)",
             "Historical reference outcome (measured): %s" % result["reference_raw"]]
    if result["reference_value"] is None:
        lines.append("Reference arithmetic: unavailable because the first reference outcome is missing or nonfinite")
    elif result["reference_value"] == 0:
        lines.append("Absolute difference: %.6f" % result["absolute_difference"])
        lines.append("Percentage improvement: unavailable because the reference outcome is zero")
    else:
        lines.append("Absolute difference: %.6f" % result["absolute_difference"])
        lines.append("Percentage improvement: %.6f%%" % result["improvement_percent"])
    lines += ["Model estimates: none; all reported outcomes are measured observations.",
              "Associations: none computed. Associations would not establish the effect of changing a setting, and correlated settings can share explanatory information.",
              "Interpretation: lower Performance- is better; this recommendation is retrospective and does not guarantee future performance or statistical superiority.",
              "Assumptions and limitations: CSV order defines the historical reference and tie-break; invalid/nonfinite outcomes are excluded from selection; no units, quality measures, resource costs, or constraints were supplied."]
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: justify.py DATASET.csv")
    result = analyze(argv[1])
    print(render(result))


if __name__ == "__main__":
    main(sys.argv)
