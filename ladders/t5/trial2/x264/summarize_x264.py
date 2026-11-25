"""Summarize X264 encoder settings and measured outcomes using stdlib only."""
import argparse
import csv
import math
import sys


MISSING = {"", "?", "NA", "N/A", "null", "None"}


def _number(value):
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None


def _stats(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None,
                "median": None, "stdev": None}
    ordered = sorted(values)
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    middle = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"count": n, "mean": mean, "min": ordered[0], "max": ordered[-1],
            "median": middle, "stdev": math.sqrt(variance)}


def summarize(path):
    """Return a JSON-serializable summary; malformed outcome cells are missing."""
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"record_count": 0, "columns": [], "outcome": None,
                    "missing_values": {}, "setting_frequencies": {}, "numeric_statistics": {},
                    "invalid_values": 0, "assumptions": "Empty CSV input."}
        header = [name.strip() for name in header]
        if not header or any(not name for name in header):
            raise ValueError("header must contain non-empty column names")
        outcome_names = [name for name in header if name == "Performance-" or name.endswith("-")]
        outcome = "Performance-" if "Performance-" in outcome_names else (outcome_names[0] if outcome_names else None)
        if outcome is None:
            raise ValueError("no measured outcome column (expected Performance- or a trailing -)")
        outcome_at = header.index(outcome)
        settings = [name for name in header if name != outcome]
        frequencies = {name: {} for name in settings}
        missing = {name: 0 for name in header}
        values = []
        invalid = 0
        records = 0
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            records += 1
            cells = row + [""] * max(0, len(header) - len(row))
            for index, name in enumerate(header):
                raw = cells[index].strip() if index < len(cells) else ""
                if raw in MISSING:
                    missing[name] += 1
                elif name != outcome:
                    frequencies[name][raw] = frequencies[name].get(raw, 0) + 1
            raw_outcome = cells[outcome_at].strip() if outcome_at < len(cells) else ""
            if raw_outcome in MISSING:
                continue
            number = _number(raw_outcome)
            if number is None:
                invalid += 1
            else:
                values.append(number)
        return {"record_count": records, "columns": header, "outcome": outcome,
                "outcome_direction": "lower is better", "missing_values": missing,
                "setting_frequencies": frequencies, "numeric_statistics": {outcome: _stats(values)},
                "invalid_values": invalid,
                "assumptions": "First row is a CSV header; cells are comma-separated; blank, ?, NA, N/A, null, and None are missing; malformed or non-finite outcomes are invalid and excluded from statistics."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        import json
        print(json.dumps(summarize(args.csv_path), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
