"""Reusable CSV summarizer and configuration-only quality predictor."""

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
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values)}

def summarize_csv(path, quality_columns=None, quality_prefixes=DEFAULT_QUALITY_PREFIXES):
    """Return a JSON-serializable summary of model-results CSV data."""
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
        selected = {name for name in columns if name.lower().startswith(
            tuple(prefix.lower() for prefix in quality_prefixes))}
    configuration = [name for name in columns if name not in selected]
    quality = {}
    for name in columns:
        if name in selected:
            valid = [_number(row.get(name, "")) for row in rows]
            quality[name] = _stats([value for value in valid if value is not None])
    settings = {}
    for name in configuration:
        raw = [row.get(name, "") for row in rows]
        numeric = [_number(value) for value in raw]
        if raw and all(value is not None for value in numeric):
            settings[name] = {"type": "numeric", "count": len(raw),
                              "unique_count": len(set(numeric)),
                              "min": min(numeric) if numeric else None,
                              "max": max(numeric) if numeric else None,
                              "values": sorted(set(numeric))}
        else:
            settings[name] = {"type": "categorical", "count": len(raw),
                              "unique_count": len(set(raw)), "values": sorted(set(raw))}
    return {"row_count": len(rows), "column_count": len(columns), "columns": columns,
            "quality_metrics": quality, "configuration": settings}

def predict_quality(path, configuration, quality_columns=None, neighbors=5):
    """Estimate quality using the closest historical configurations.

    Numeric settings are range-normalized; categorical settings match exactly.
    The estimate is a distance-weighted average and is intended as a quick
    pre-evaluation forecast, not a replacement for full evaluation.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [name.strip() for name in (reader.fieldnames or [])]
        rows = list(reader)
    if not columns:
        raise ValueError("CSV must contain a header")
    quality = set(quality_columns or ())
    if not quality:
        quality = {name for name in columns if name.lower().startswith(DEFAULT_QUALITY_PREFIXES)}
    config_columns = [name for name in columns if name not in quality]
    missing = set(configuration).difference(config_columns)
    if missing:
        raise ValueError("Unknown configuration columns: " + ", ".join(sorted(missing)))
    spans = {}
    for name in config_columns:
        vals = [_number(row.get(name, "")) for row in rows]
        valid = [v for v in vals if v is not None]
        spans[name] = (min(valid), max(valid)) if valid else None
    def distance(row):
        total = 0.0
        for name in config_columns:
            target = configuration.get(name, row.get(name, ""))
            actual = row.get(name, "")
            target_num, actual_num = _number(target), _number(actual)
            if target_num is not None and actual_num is not None and spans[name]:
                low, high = spans[name]
                total += 0.0 if high == low else abs(target_num - actual_num) / (high - low)
            else:
                total += 0.0 if str(target) == str(actual) else 1.0
        return total
    ranked = sorted(((distance(row), row) for row in rows), key=lambda item: item[0])[:neighbors]
    result = {name: None for name in sorted(quality)}
    for name in result:
        weighted = [(1.0 / (dist + 1e-12), _number(row.get(name, ""))) for dist, row in ranked]
        weighted = [(weight, value) for weight, value in weighted if value is not None]
        if weighted:
            result[name] = sum(weight * value for weight, value in weighted) / sum(weight for weight, _ in weighted)
    return {"configuration": dict(configuration), "neighbors_used": len(ranked), "predicted_quality": result}
