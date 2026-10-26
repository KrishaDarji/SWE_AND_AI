#!/usr/bin/env python3
"""Create an evidence-based project estimate and sponsor-facing justification."""
import argparse, csv, math, statistics, sys

def read_records(path):
    with open(path, newline="", encoding="utf-8") as handle:
        records = []
        for row in csv.DictReader(handle):
            try:
                values = (float(row["Kloc+"]), float(row["Effort-"]), float(row["MonthsX"]))
            except (KeyError, TypeError, ValueError):
                continue
            if values[0] > 0 and values[1] >= 0 and values[2] > 0: records.append(values)
    if not records: raise ValueError("no usable Kloc+, Effort-, and MonthsX records")
    return records

def percentile(values, fraction):
    ordered = sorted(values); position = (len(ordered)-1)*fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)

def estimate(records, target_kloc):
    distances = sorted(abs(row[0]-target_kloc) for row in records)
    window = max(5, min(len(records), math.ceil(len(records)*.25)))
    cutoff = distances[window-1]
    peers = [row for row in records if abs(row[0]-target_kloc) <= cutoff]
    efforts, months = [row[1] for row in peers], [row[2] for row in peers]
    return {"target_kloc":target_kloc, "peer_count":len(peers), "dataset_count":len(records),
            "effort_median":statistics.median(efforts), "effort_low":percentile(efforts,.25),
            "effort_high":percentile(efforts,.75), "months_median":statistics.median(months),
            "months_low":percentile(months,.25), "months_high":percentile(months,.75)}

def render(result):
    return (f"Estimate for {result['target_kloc']:.1f} KLOC: {result['effort_median']:.1f} person-months over {result['months_median']:.1f} months.\n"
            f"Recommended planning range: {result['effort_low']:.1f}-{result['effort_high']:.1f} person-months and {result['months_low']:.1f}-{result['months_high']:.1f} months.\n\n"
            f"Why this is defensible: NASA93 has {result['dataset_count']} usable projects; this uses the median of {result['peer_count']} projects nearest in size. The median limits outliers, and the interquartile range makes uncertainty visible.\n\n"
            "Sponsor decision: fund to the midpoint estimate, retain the planning range as contingency, and revisit after requirements and team capability are validated. Historical evidence is not a guarantee; scope, staffing, complexity, and delivery practices can move the result outside it.")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path"); parser.add_argument("--kloc", type=float, required=True)
    args = parser.parse_args(argv)
    if args.kloc <= 0: parser.error("--kloc must be positive")
    try: print(render(estimate(read_records(args.csv_path), args.kloc)))
    except (OSError, ValueError) as exc: print(f"error: {exc}", file=sys.stderr); return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())

