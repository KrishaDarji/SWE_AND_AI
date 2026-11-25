#!/usr/bin/env python3
"""Standard-library statistical certification for paired model measurements."""
import argparse
import csv
import math
import statistics
from typing import Dict, Iterable, List


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def certify(baseline: Iterable[float], candidate: Iterable[float],
            higher_is_better: bool = True, alpha: float = 0.05) -> Dict[str, float]:
    """Certify a paired change using a normal approximation to the t statistic.

    ``candidate`` and ``baseline`` must be aligned repeated measurements.  The
    returned improvement is positive when the candidate is better, including
    for error metrics when ``higher_is_better`` is False.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    old, new = list(map(float, baseline)), list(map(float, candidate))
    if len(old) != len(new) or len(old) < 2:
        raise ValueError("paired samples must have equal length of at least 2")
    if not all(math.isfinite(x) for x in old + new):
        raise ValueError("samples must contain only finite numbers")
    sign = 1.0 if higher_is_better else -1.0
    differences = [sign * (b - a) for a, b in zip(old, new)]
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences)
    se = sd / math.sqrt(len(differences))
    z = mean / se if se else (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0)
    p = 2.0 * (1.0 - _normal_cdf(abs(z))) if math.isfinite(z) else 0.0
    critical = 1.959963984540054 if alpha == 0.05 else _normal_quantile(1 - alpha / 2)
    low, high = mean - critical * se, mean + critical * se
    return {"n": len(differences), "mean_improvement": mean,
            "std_improvement": sd, "ci_low": low, "ci_high": high,
            "p_value": p, "meaningful": float(p < alpha and low > 0)}


def _normal_quantile(p: float) -> float:
    # Acklam's rational approximation; adequate for confidence thresholds.
    if not 0 < p < 1:
        raise ValueError("probability must be between 0 and 1")
    a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    q = p - .5
    if abs(q) <= .425:
        r = .180625 - q*q
        return q * sum(a[i] * r**i for i in range(6)) / (1 + sum(b[i] * r**(i+1) for i in range(5)))
    r = math.sqrt(-math.log(min(p, 1-p)))
    x = sum(c[i] * r**i for i in range(6)) / (1 + sum(d[i] * r**(i+1) for i in range(4)))
    return x if q > 0 else -x


def read_column(path: str, name: str) -> List[float]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [float(row[name]) for row in csv.DictReader(handle) if row.get(name, "") != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify paired model improvement")
    parser.add_argument("csv_path")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    result = certify(read_column(args.csv_path, args.baseline), read_column(args.csv_path, args.candidate), not args.lower_is_better, args.alpha)
    for key, value in result.items():
        print(f"{key}={value:.10g}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
