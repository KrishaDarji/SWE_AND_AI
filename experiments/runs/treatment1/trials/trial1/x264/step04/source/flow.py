#!/usr/bin/env python3
"""Evaluate x264 benchmark configurations as results arrive one at a time."""
import argparse
import json
import math
import sys


class FlowEvaluator:
    """Online lower-is-better evaluator for benchmark results.

    A recommendation is sufficient when at least ``min_results`` results have
    arrived and the best configuration's confidence interval is separated from
    the runner-up's interval by ``margin`` percent. With one observation per
    configuration, intervals are unavailable, so the evaluator conservatively
    waits until a repeated configuration provides an estimate of noise.
    """

    def __init__(self, min_results=5, margin=1.0):
        if min_results < 1 or margin < 0:
            raise ValueError("min_results must be positive and margin non-negative")
        self.min_results = min_results
        self.margin = float(margin)
        self._observations = {}
        self._count = 0

    def add(self, configuration, measurement):
        """Add one benchmark result and return the current evaluation."""
        if not isinstance(configuration, dict) or not configuration:
            raise ValueError("configuration must be a non-empty dictionary")
        try:
            value = float(measurement)
        except (TypeError, ValueError) as exc:
            raise ValueError("measurement must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError("measurement must be finite")
        key = tuple(sorted((str(k), str(v)) for k, v in configuration.items()))
        record = self._observations.setdefault(key, {"configuration": dict(configuration), "values": []})
        record["values"].append(value)
        self._count += 1
        return self.summary()

    @staticmethod
    def _stats(values):
        mean = sum(values) / len(values)
        if len(values) < 2:
            return mean, None
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, math.sqrt(variance / len(values))

    def summary(self):
        ranked = []
        for record in self._observations.values():
            mean, standard_error = self._stats(record["values"])
            ranked.append({"configuration": record["configuration"], "mean": mean,
                           "observations": len(record["values"]), "standard_error": standard_error})
        ranked.sort(key=lambda item: item["mean"])
        sufficient = False
        reason = "need more benchmark results"
        if self._count >= self.min_results and len(ranked) >= 2:
            best, runner = ranked[0], ranked[1]
            if best["standard_error"] is None or runner["standard_error"] is None:
                reason = "need repeated measurements for uncertainty estimates"
            else:
                separation = runner["mean"] - best["mean"]
                uncertainty = 2.0 * (best["standard_error"] + runner["standard_error"])
                required = abs(best["mean"]) * self.margin / 100.0
                sufficient = separation > max(uncertainty, required)
                reason = "separation exceeds uncertainty and requested margin" if sufficient else "best and runner-up overlap or margin is small"
        return {"results": self._count, "configurations": len(ranked),
                "recommended": ranked[0] if ranked else None, "ranking": ranked,
                "sufficient": sufficient, "reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-results", type=int, default=5)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()
    evaluator = FlowEvaluator(args.min_results, args.margin)
    for line in sys.stdin:
        if line.strip():
            item = json.loads(line)
            print(json.dumps(evaluator.add(item["configuration"], item["measurement"]), sort_keys=True))


if __name__ == "__main__":
    main()
