#!/usr/bin/env python3
"""Spot unusual model-quality values in a CSV using only the standard library."""
import argparse, csv, json, math, statistics
from collections import defaultdict

QUALITY = {"MRE-": "low", "ACC+": "high", "PRED40+": "high"}

def _robust_flags(values):
    med = statistics.median(values)
    mad = statistics.median(abs(x - med) for x in values)
    if mad:
        return {i for i, x in enumerate(values) if abs(x - med) > 3.5 * 1.4826 * mad}, med, mad
    if len(values) < 2:
        return set(), med, mad
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    if not iqr:
        return set(), med, mad
    return {i for i, x in enumerate(values) if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr}, med, mad

def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV has no data rows")
    missing = [c for c in QUALITY if c not in rows[0]]
    if missing:
        raise ValueError("missing quality columns: " + ", ".join(missing))
    parsed, errors = [], []
    for line, original in enumerate(rows, 2):
        row = dict(original)
        bad = False
        for col in QUALITY:
            try:
                row[col] = float(original[col])
                if not math.isfinite(row[col]):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append({"row": line, "column": col, "value": original.get(col)})
                bad = True
        if not bad:
            parsed.append((line, row))
    flags, stats = defaultdict(list), {}
    for col in QUALITY:
        values = [r[col] for _, r in parsed]
        indexes, median, mad = _robust_flags(values) if values else (set(), None, None)
        stats[col] = {"median": median, "mad": mad, "outlier_count": len(indexes)}
        for i in indexes:
            flags[parsed[i][0]].append(col + " robust outlier")
    configs = defaultdict(list)
    for line, row in parsed:
        configs[tuple((k, row[k]) for k in row if k not in QUALITY)].append((line, row))
    for group in configs.values():
        if len(group) > 1 and any(max(r[c] for _, r in group) != min(r[c] for _, r in group) for c in QUALITY):
            for line, _ in group:
                flags[line].append("duplicate configuration has conflicting quality")
    for line, row in parsed:
        if row["MRE-"] < 0 or not (0 <= row["ACC+"] <= 1) or not (0 <= row["PRED40+"] <= 1):
            flags[line].append("quality value outside expected range")
    flagged = [{"row": line, "reasons": reasons} for line, reasons in sorted(flags.items())]
    conclusion = "Potential anomalies found; inspect flagged rows and measurement provenance." if flagged or errors else "No unusual quality result detected by these checks."
    return {"rows": len(rows), "valid_rows": len(parsed), "parse_errors": errors, "statistics": stats, "flagged": flagged, "conclusion": conclusion}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    print(json.dumps(analyze(parser.parse_args().csv_path), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
