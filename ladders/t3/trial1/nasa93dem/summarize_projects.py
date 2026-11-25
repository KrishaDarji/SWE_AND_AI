#!/usr/bin/env python3
"""Summarize NASA93 historical projects and delivery outcomes."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


OUTCOME_FIELDS = {"Kloc+": "kloc", "Effort-": "effort", "Defects-": "defects", "MonthsX": "months"}
PREDICTORS = ["Kloc+", "YearX", "centerX", "rely", "data", "cplx", "time", "stor", "acap", "pcap", "apex", "plex", "ltex"]
ORDINAL = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no rows")
    required = set(OUTCOME_FIELDS) | set(PREDICTORS)
    missing = required - set(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    return rows


def number(value):
    return float(value.strip())


def value(row, field):
    raw = row[field].strip().lower()
    try:
        return float(raw)
    except ValueError:
        if raw not in ORDINAL:
            raise ValueError("unsupported categorical value %r in %s" % (raw, field))
        return float(ORDINAL[raw])


def stats(values):
    ordered = sorted(values)
    return {"min": round(min(values), 3), "median": round(statistics.median(values), 3),
            "mean": round(statistics.mean(values), 3), "max": round(max(values), 3),
            "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "p90": round(ordered[min(len(ordered) - 1, math.ceil(.9 * len(ordered)) - 1)], 3)}


def summarize(rows):
    outcomes = {name: stats([number(row[col]) for row in rows]) for col, name in OUTCOME_FIELDS.items()}
    by_year = {}
    for row in rows:
        by_year.setdefault(row["YearX"].strip(), []).append(row)
    yearly = []
    for year in sorted(by_year, key=float):
        group = by_year[year]
        yearly.append({"year": int(float(year)), "projects": len(group),
                       "median_months": round(statistics.median(number(x["MonthsX"]) for x in group), 3),
                       "median_effort": round(statistics.median(number(x["Effort-"]) for x in group), 3),
                       "median_defects": round(statistics.median(number(x["Defects-"]) for x in group), 3)})

    features = [[value(row, field) for field in PREDICTORS] for row in rows]
    scaled = StandardScaler().fit_transform(features)
    model = LinearRegression().fit(scaled, [number(row["MonthsX"]) for row in rows])
    coefficients = sorted(({"feature": name, "coefficient": round(float(coef), 4)}
                           for name, coef in zip(PREDICTORS, model.coef_)),
                          key=lambda item: abs(item["coefficient"]), reverse=True)
    return {"projects": len(rows), "outcomes": outcomes, "by_year": yearly,
            "schedule_model": {"r_squared": round(float(model.score(scaled, [number(row["MonthsX"]) for row in rows])), 4),
                                "standardized_coefficients": coefficients}}


def render(summary):
    o = summary["outcomes"]
    lines = ["Historical NASA93 project summary", "Projects: %d" % summary["projects"], "",
             "Delivery outcomes (min / median / mean / p90 / max):"]
    for key in ("kloc", "effort", "defects", "months"):
        s = o[key]
        lines.append("- %s: %.3f / %.3f / %.3f / %.3f / %.3f" % (key, s["min"], s["median"], s["mean"], s["p90"], s["max"]))
    lines += ["", "Yearly median outcomes:"]
    for item in summary["by_year"]:
        lines.append("- %d (%d projects): months %.3f, effort %.3f, defects %.3f" % (item["year"], item["projects"], item["median_months"], item["median_effort"], item["median_defects"]))
    lines += ["", "Standardized schedule associations (linear model; descriptive, not causal):"]
    lines.extend("- %s: %.4f" % (x["feature"], x["coefficient"]) for x in summary["schedule_model"]["standardized_coefficients"][:5])
    lines.append("Model R-squared: %.4f" % summary["schedule_model"]["r_squared"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = summarize(read_rows(args.dataset))
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else render(result), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
