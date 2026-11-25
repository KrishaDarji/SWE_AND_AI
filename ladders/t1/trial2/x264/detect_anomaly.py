#!/usr/bin/env python3
"""Detect benchmark measurements inconsistent with nearby configurations."""
import argparse
import csv
import math
import statistics
import sys


def median(values):
    return statistics.median(values)


def read_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no data rows")
    target = "Performance-"
    if target not in rows[0]:
        raise ValueError("missing Performance- column")
    features = [name for name in rows[0] if name != target]
    parsed = []
    for number, row in enumerate(rows, 2):
        try:
            bits = tuple(int(row[name]) for name in features)
            value = float(row[target].strip())
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid row %d: %s" % (number, exc)) from exc
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("non-binary feature in row %d" % number)
        parsed.append((bits, value, number))
    return features, parsed


def analyze(path, threshold=6.0):
    features, rows = read_rows(path)
    lookup = {bits: (value, number) for bits, value, number in rows}
    residuals = []
    records = []
    for bits, value, number in rows:
        neighbors = []
        for index in range(len(bits)):
            candidate = list(bits)
            candidate[index] = 1 - candidate[index]
            if tuple(candidate) in lookup:
                neighbors.append(lookup[tuple(candidate)][0])
        if not neighbors:
            prediction = value
            residual = 0.0
        else:
            prediction = median(neighbors)
            residual = value - prediction
        residuals.append(residual)
        records.append({"row": number, "value": value, "prediction": prediction,
                        "residual": residual, "neighbors": len(neighbors)})

    center = median(residuals)
    deviations = [abs(item - center) for item in residuals]
    mad = median(deviations)
    scale = 1.4826 * mad
    if scale == 0:
        scale = statistics.pstdev(residuals) or 1.0
    for record in records:
        score = abs(record["residual"] - center) / scale
        record["score"] = score
        record["anomalous"] = score >= threshold
    return features, records, center, mad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args(argv)
    features, records, center, mad = analyze(args.dataset, args.threshold)
    flagged = [record for record in records if record["anomalous"]]
    print("rows=%d features=%d residual_median=%.6f mad=%.6f" %
          (len(records), len(features), center, mad))
    print("anomalies=%d" % len(flagged))
    for record in sorted(flagged, key=lambda item: item["score"], reverse=True)[:10]:
        print("row=%d value=%.6f predicted=%.6f residual=%.6f score=%.3f neighbors=%d" %
              (record["row"], record["value"], record["prediction"],
               record["residual"], record["score"], record["neighbors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
