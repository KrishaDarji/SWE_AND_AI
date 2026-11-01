"""Summarize historical X264 encoder settings and measured outcomes."""
import csv, math, statistics, sys

MISSING = {"", "?", "nil", "NA", "N/A", "null"}

def _number(value):
    text = value.strip()
    if text in MISSING:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None

def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return [], [], []
    header = [cell.strip() for cell in rows[0]]
    records = []
    invalid = []
    for line, row in enumerate(rows[1:], 2):
        if not any(cell.strip() for cell in row):
            continue
        values = row + [""] * max(0, len(header) - len(row))
        values = values[:len(header)]
        records.append(dict(zip(header, values)))
    return header, records, invalid

def summarize(path):
    header, records, _ = load(path)
    outcome_names = [name for name in header if name.endswith("-")]
    outcome = outcome_names[0] if outcome_names else None
    settings = [name for name in header if name != outcome]
    frequencies = {}
    for name in settings:
        counts = {}
        for record in records:
            value = record.get(name, "").strip()
            key = value if value not in MISSING else "<missing>"
            counts[key] = counts.get(key, 0) + 1
        frequencies[name] = counts
    outcomes = []
    missing = 0
    invalid = 0
    if outcome:
        for record in records:
            raw = record.get(outcome, "")
            value = _number(raw)
            if raw.strip() in MISSING:
                missing += 1
            elif value is None:
                invalid += 1
            else:
                outcomes.append(value)
    stats = None
    if outcomes:
        stats = {"count": len(outcomes), "min": min(outcomes),
                 "max": max(outcomes), "mean": statistics.fmean(outcomes),
                 "median": statistics.median(outcomes),
                 "stdev": statistics.stdev(outcomes) if len(outcomes) > 1 else 0.0}
    return {"record_count": len(records), "headers": header, "settings": settings,
            "outcome": outcome, "direction": "lower is better" if outcome else None,
            "frequencies": frequencies, "outcome_stats": stats,
            "missing_outcomes": missing, "invalid_outcomes": invalid,
            "missing_input": not bool(header)}

def render(summary):
    lines = ["Historical X264 run summary", "record_count: %d" % summary["record_count"],
             "outcome: %s (%s)" % (summary["outcome"] or "none", summary["direction"] or "unavailable")]
    if summary["outcome_stats"]:
        lines.append("outcome_stats: " + ", ".join("%s=%.6g" % item for item in summary["outcome_stats"].items()))
    else:
        lines.append("outcome_stats: unavailable")
    lines.append("missing_outcomes: %d; invalid_outcomes: %d" % (summary["missing_outcomes"], summary["invalid_outcomes"]))
    lines.append("setting_frequencies:")
    for name, counts in summary["frequencies"].items():
        lines.append("  %s: %s" % (name, ", ".join("%s=%d" % pair for pair in sorted(counts.items()))))
    return "\n".join(lines)

def main(argv):
    if len(argv) != 2:
        print("usage: x264_summary.py DATASET", file=sys.stderr); return 2
    try:
        print(render(summarize(argv[1])))
        return 0
    except (OSError, csv.Error) as exc:
        print("error: %s" % exc, file=sys.stderr); return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
