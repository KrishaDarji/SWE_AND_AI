"""Reusable CSV summarizer for model-quality metrics and configuration settings."""

import csv
import math
import statistics


DEFAULT_QUALITY_PREFIXES = ("mre", "acc", "pred")


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def summarize_csv(path, quality_columns=None, quality_prefixes=DEFAULT_QUALITY_PREFIXES):
    """Return a JSON-serializable summary of a model-results CSV file.

    Quality columns may be supplied explicitly; otherwise columns whose lower-case
    names start with a configured prefix are quality metrics. Remaining columns are
    treated as model configuration settings.
    """
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        columns = [name.strip() for name in reader.fieldnames]
        if any(not name for name in columns):
            raise ValueError("CSV contains an empty column name")
        rows = list(reader)

    selected = set(quality_columns or ())
    unknown = selected.difference(columns)
    if unknown:
        raise ValueError("Unknown quality columns: " + ", ".join(sorted(unknown)))
    if not selected:
        selected = {
            name for name in columns
            if name.lower().startswith(tuple(prefix.lower() for prefix in quality_prefixes))
        }
    configuration = [name for name in columns if name not in selected]
    quality = {}
    for name in columns:
        if name in selected:
            values = [_number(row.get(name, "")) for row in rows]
            valid = [value for value in values if value is not None]
            quality[name] = _stats(valid)

    settings = {}
    for name in configuration:
        raw = [row.get(name, "") for row in rows]
        numeric = [_number(value) for value in raw]
        if raw and all(value is not None for value in numeric):
            settings[name] = {
                "type": "numeric",
                "count": len(raw),
                "unique_count": len(set(numeric)),
                "min": min(numeric) if numeric else None,
                "max": max(numeric) if numeric else None,
                "values": sorted(set(numeric)),
            }
        else:
            settings[name] = {
                "type": "categorical",
                "count": len(raw),
                "unique_count": len(set(raw)),
                "values": sorted(set(raw)),
            }
    return {
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "quality_metrics": quality,
        "configuration": settings,
    }
