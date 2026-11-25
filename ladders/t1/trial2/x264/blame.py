#!/usr/bin/env python3
"""Explain which X264 settings are associated with slow encodes."""
import argparse
import csv
import math
import statistics
import sys


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze(path, slow_fraction=0.25):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    target = "Performance-"
    if target not in rows[0]:
        raise ValueError("missing Performance- column")
    settings = [name for name in rows[0] if name != target]
    runtimes = [float(row[target]) for row in rows]
    threshold = percentile(runtimes, 1 - slow_fraction)
    slow = [value >= threshold for value in runtimes]
    results = []
    for setting in settings:
        enabled = [i for i, row in enumerate(rows) if row[setting] == "1"]
        disabled = [i for i, row in enumerate(rows) if row[setting] == "0"]
        if not enabled or not disabled:
            continue
        on = [runtimes[i] for i in enabled]
        off = [runtimes[i] for i in disabled]
        on_slow = sum(slow[i] for i in enabled) / len(enabled)
        off_slow = sum(slow[i] for i in disabled) / len(disabled)
        delta = statistics.mean(on) - statistics.mean(off)
        scale = statistics.pstdev(runtimes) or 1.0
        correlation = point_biserial([row[setting] == "1" for row in rows], runtimes)
        results.append({
            "setting": setting, "enabled": len(enabled), "disabled": len(disabled),
            "mean_enabled": statistics.mean(on), "mean_disabled": statistics.mean(off),
            "delta": delta, "slow_rate_enabled": on_slow, "slow_rate_disabled": off_slow,
            "slow_rate_lift": on_slow - off_slow, "correlation": correlation,
            "score": (delta / scale) + (on_slow - off_slow),
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"rows": len(rows), "settings": len(settings), "threshold": threshold,
            "slow_fraction": slow_fraction, "results": results,
            "runtime_min": min(runtimes), "runtime_max": max(runtimes),
            "runtime_mean": statistics.mean(runtimes)}


def point_biserial(enabled, values):
    one = [value for flag, value in zip(enabled, values) if flag]
    zero = [value for flag, value in zip(enabled, values) if not flag]
    if not one or not zero:
        return 0.0
    p = len(one) / len(values)
    q = 1 - p
    return (statistics.mean(one) - statistics.mean(zero)) * math.sqrt(p * q) / (statistics.pstdev(values) or 1.0)


def render(report, limit=10):
    lines = [
        "X264 slow-encode blame report",
        "================================",
        f"Rows: {report['rows']} | settings: {report['settings']}",
        f"Runtime: min={report['runtime_min']:.3f}, mean={report['runtime_mean']:.3f}, max={report['runtime_max']:.3f}",
        f"Slow threshold (top {report['slow_fraction']:.0%}): {report['threshold']:.3f}",
        "",
        "Ranked by standardized runtime delta + slow-rate lift (positive means slower when enabled):",
        "setting | mean_on | mean_off | delta | slow_lift | correlation | score",
    ]
    for item in report["results"][:limit]:
        lines.append(f"{item['setting']} | {item['mean_enabled']:.3f} | {item['mean_disabled']:.3f} | "
                     f"{item['delta']:+.3f} | {item['slow_rate_lift']:+.3f} | "
                     f"{item['correlation']:+.3f} | {item['score']:+.3f}")
    lines += ["", "Interpretation: positive-ranked settings are associated with slower encodes in this dataset; this is not a causal proof."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    print(render(analyze(args.dataset), args.limit))


if __name__ == "__main__":
    main()
