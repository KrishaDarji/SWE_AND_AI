"""Rank X264 settings by their marginal association with Performance-."""
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

def _setting(value):
    value = (value or "").strip()
    if value in ("", "?"):
        return None
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
    except ValueError:
        pass
    return value

def _summary(values):
    ordered = sorted(values)
    n = len(ordered)
    return {"count": n, "mean": sum(ordered) / n if n else None,
            "min": ordered[0] if n else None, "median": ordered[(n - 1) // 2] if n else None,
            "max": ordered[-1] if n else None}

def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if TARGET not in fields:
            raise ValueError("dataset must contain Performance-")
        settings = [field for field in fields if field != TARGET]
        rows = []
        for row in reader:
            outcome = _number(row.get(TARGET, ""))
            if outcome is not None:
                rows.append((row, outcome))
        return settings, rows

def _score(groups, overall):
    total = sum((value - overall) ** 2 for values in groups.values() for value in values)
    between = sum(len(values) * (sum(values) / len(values) - overall) ** 2
                  for values in groups.values() if values)
    return between / total if total else 0.0

def blame(path):
    settings, rows = _read(path)
    if not rows:
        raise ValueError("at least one finite Performance- value is required")
    overall = sum(value for _, value in rows) / len(rows)
    ranking = []
    for name in settings:
        groups = {}
        missing = 0
        for row, outcome in rows:
            value = _setting(row.get(name))
            if value is None:
                missing += 1
            else:
                groups.setdefault(value, []).append(outcome)
        score = _score(groups, overall) if len(groups) > 1 else 0.0
        levels = [{"value": value, "summary": _summary(groups[value])}
                  for value in sorted(groups)]
        ranking.append({"setting": name, "score": score, "level_count": len(levels),
                        "valid_count": sum(len(v) for v in groups.values()),
                        "missing_count": missing, "levels": levels,
                        "direction": "higher level mean indicates slower observed encodes"
                        if len(levels) > 1 else "not estimable from multiple levels"})
    ranking.sort(key=lambda item: (-item["score"], item["setting"]))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    return {"target": TARGET, "direction": "lower Performance- is better",
            "score": "eta_squared = between-group sum of squares / total sum of squares",
            "rows_with_finite_outcome": len(rows), "overall_summary": _summary([v for _, v in rows]),
            "ranking": ranking,
            "interpretation": "marginal observational association, not causal responsibility; correlated settings can share information"}

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("usage: x264_blame.py DATASET", file=sys.stderr)
        return 2
    print(json.dumps(blame(argv[0]), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
