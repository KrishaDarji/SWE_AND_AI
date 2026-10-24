#!/usr/bin/env python3
"""Certify whether the fastest observed x264 configuration beats noise."""
import argparse
import csv
import json
import math
from pathlib import Path


def _mad(values):
    values = sorted(values)
    if not values:
        return 0.0
    middle = values[len(values) // 2] if len(values) % 2 else (values[len(values)//2-1] + values[len(values)//2]) / 2
    deviations = sorted(abs(value - middle) for value in values)
    return deviations[len(deviations)//2] if len(deviations) % 2 else (deviations[len(deviations)//2-1] + deviations[len(deviations)//2]) / 2


def certify(dataset, measure="Performance-", neighbors=5, threshold=2.0):
    """Return a noise-aware certification for lower-is-faster measurements.

    Noise is estimated from leave-one-out nearest-neighbor residuals.  The
    result is meaningful only as a screening test when configurations are not
    replicated; replicated measurements should be preferred.
    """
    if neighbors < 1 or threshold <= 0:
        raise ValueError("neighbors and threshold must be positive")
    with Path(dataset).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or measure not in rows[0]:
        raise ValueError("dataset is empty or measure column is missing")
    fields = [field for field in rows[0] if field != measure]
    points = []
    for row in rows:
        try:
            points.append((tuple(float(row[field]) for field in fields), float(row[measure]), row))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("non-numeric setting or measurement") from exc
    ranges = [max(p[0][i] for p in points) - min(p[0][i] for p in points) for i in range(len(fields))]
    ranges = [value if value else 1.0 for value in ranges]

    residuals = []
    for index, (point, value, _) in enumerate(points):
        distances = []
        for other, other_value, _ in points:
            if other is point:
                continue
            distance = math.sqrt(sum(((a-b)/scale) ** 2 for a, b, scale in zip(point, other, ranges)))
            distances.append((distance, other_value))
        distances.sort()
        chosen = distances[:min(neighbors, len(distances))]
        prediction = sum(item[1] for item in chosen) / len(chosen)
        residuals.append(value - prediction)
    noise = 1.4826 * _mad(residuals)
    ranked = sorted(points, key=lambda item: item[1])
    best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None
    improvement = runner_up[1] - best[1] if runner_up else 0.0
    meaningful = runner_up is not None and improvement > threshold * noise
    return {"best_configuration": dict(zip(fields, best[2].values())), "best_measurement": best[1],
            "runner_up_measurement": runner_up[1] if runner_up else None, "improvement": improvement,
            "noise_scale": noise, "threshold": threshold, "meaningfully_faster": meaningful,
            "noise_method": "1.4826 * MAD of leave-one-out nearest-neighbor residuals",
            "rows": len(rows), "unique_configurations": len({p[0] for p in points}), "settings": fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-k", "--neighbors", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(certify(args.dataset, neighbors=args.neighbors), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
