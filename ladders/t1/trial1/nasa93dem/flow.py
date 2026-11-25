#!/usr/bin/env python3
"""Evaluate effort estimates online as project measurements arrive."""
import argparse
import csv
import json
import math
import statistics
import sys


class FlowEvaluator:
    """Track one-at-a-time effort estimates and their observed accuracy."""

    def __init__(self, tolerance=0.25):
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = float(tolerance)
        self.records = []

    def add(self, project, estimate, actual):
        if not project:
            raise ValueError("project is required")
        try:
            estimate, actual = float(estimate), float(actual)
        except (TypeError, ValueError) as exc:
            raise ValueError("estimate and actual must be numeric") from exc
        if not math.isfinite(estimate) or not math.isfinite(actual) or estimate < 0 or actual <= 0:
            raise ValueError("estimate must be non-negative and actual must be positive")
        error = actual - estimate
        relative_error = abs(error) / actual
        self.records.append({"project": str(project), "estimate": estimate,
                             "actual": actual, "relative_error": relative_error,
                             "within_tolerance": relative_error <= self.tolerance})
        errors = [r["relative_error"] for r in self.records]
        return {"project": str(project), "observations": len(self.records),
                "mean_relative_error": statistics.fmean(errors),
                "median_relative_error": statistics.median(errors),
                "within_tolerance": self.records[-1]["within_tolerance"],
                "pass_rate": sum(r["within_tolerance"] for r in self.records) / len(errors)}

    def summary(self):
        if not self.records:
            return {"observations": 0, "mean_relative_error": None, "pass_rate": None}
        errors = [r["relative_error"] for r in self.records]
        return {"observations": len(errors), "mean_relative_error": statistics.fmean(errors),
                "median_relative_error": statistics.median(errors),
                "pass_rate": sum(r["within_tolerance"] for r in self.records) / len(errors)}


def run_csv(path, tolerance):
    evaluator = FlowEvaluator(tolerance)
    historical = []
    output = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            actual = float(row["Effort-"])
            # Expanding historical mean is deliberately based only on prior projects.
            estimate = statistics.fmean(historical) if historical else actual
            output.append(evaluator.add(row["idX"], estimate, actual))
            historical.append(actual)
    return output, evaluator.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--tolerance", type=float, default=0.25)
    args = parser.parse_args(argv)
    results, summary = run_csv(args.csv_path, args.tolerance)
    print(json.dumps({"projects": len(results), "final": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
