"""Reusable CSV summarizer for model-quality metrics and configuration settings."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _summarize_numeric(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def summarize_rows(rows, quality_columns=None, configuration_columns=None):
    """Return a JSON-serializable summary from CSV row dictionaries.

    Column roles are explicit and reusable: callers can pass any quality and
    configuration column names, including columns introduced by later skills.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("CSV contains no data rows")
    columns = list(rows[0])
    quality_columns = list(quality_columns or columns[:3])
    configuration_columns = list(configuration_columns or columns[3:])
    missing = [c for c in quality_columns + configuration_columns if c not in columns]
    if missing:
        raise ValueError("Unknown columns: " + ", ".join(missing))

    quality = {}
    for column in quality_columns:
        values = [_number(row[column]) for row in rows]
        usable = [v for v in values if v is not None]
        quality[column] = _summarize_numeric(usable) if usable else {"count": 0}

    configuration = {}
    for column in configuration_columns:
        values = [row[column] for row in rows]
        numeric = [_number(v) for v in values]
        if all(v is not None for v in numeric):
            configuration[column] = {"kind": "numeric", **_summarize_numeric(numeric)}
        else:
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            configuration[column] = {
                "kind": "categorical", "count": len(values),
                "unique": len(counts), "values": dict(sorted(counts.items())),
            }
    return {"row_count": len(rows), "quality_metrics": quality,
            "configuration": configuration}


def summarize_csv(path, quality_columns=None, configuration_columns=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        return summarize_rows(reader, quality_columns, configuration_columns)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--quality", nargs="+", help="quality metric columns")
    parser.add_argument("--configuration", nargs="+", help="configuration columns")
    args = parser.parse_args(argv)
    print(json.dumps(summarize_csv(args.csv_path, args.quality, args.configuration),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
