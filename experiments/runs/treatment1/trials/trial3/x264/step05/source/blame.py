#!/usr/bin/env python3
"""Explain which binary x264 settings are associated with slow encodes."""

import argparse
import csv
import statistics
import sys


def read_measurements(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no measurements")
    fields = rows[0].keys()
    target = next((f for f in fields if f.strip().lower() in ("performance-", "performance", "encode_time", "time")), None)
    if target is None:
        raise ValueError("could not find a performance column")
    settings = [f for f in fields if f != target]
    data = []
    for number, row in enumerate(rows, 2):
        try:
            value = float(row[target].strip())
            bits = {name: int(row[name].strip()) for name in settings}
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("invalid row %d: %s" % (number, exc))
        if any(bit not in (0, 1) for bit in bits.values()):
            raise ValueError("settings must be binary (row %d)" % number)
        data.append((bits, value))
    return settings, data, target


def rank_settings(settings, data):
    overall = statistics.mean(value for _, value in data)
    ranked = []
    for setting in settings:
        on = [value for bits, value in data if bits[setting] == 1]
        off = [value for bits, value in data if bits[setting] == 0]
        if not on or not off:
            continue
        on_mean, off_mean = statistics.mean(on), statistics.mean(off)
        ranked.append({
            "setting": setting,
            "on_mean": on_mean,
            "off_mean": off_mean,
            "delta": on_mean - off_mean,
            "relative_percent": 100.0 * (on_mean - off_mean) / off_mean if off_mean else 0.0,
            "on_count": len(on), "off_count": len(off),
        })
    ranked.sort(key=lambda item: item["delta"], reverse=True)
    return overall, ranked


def render(path, top=10):
    settings, data, target = read_measurements(path)
    overall, ranked = rank_settings(settings, data)
    slowest = sorted(data, key=lambda pair: pair[1], reverse=True)[:top]
    lines = ["x264 slow-encode blame", "", "Measurements: %d" % len(data),
             "Performance column: %s" % target, "Overall mean: %.3f" % overall,
             "", "Settings associated with slower encodes (enabled mean minus disabled mean):"]
    for item in ranked:
        lines.append("- %s: %+0.3f (%+0.2f%%); enabled %.3f vs disabled %.3f (n=%d/%d)" %
                     (item["setting"], item["delta"], item["relative_percent"],
                      item["on_mean"], item["off_mean"], item["on_count"], item["off_count"]))
    lines += ["", "Slowest observed configurations:"]
    for bits, value in slowest:
        enabled = [name for name in settings if bits[name]]
        lines.append("- %.3f: %s" % (value, ", ".join(enabled) if enabled else "none"))
    lines += ["", "Interpretation: these are unadjusted associations in this controlled measurement table, not causal effects; correlated or one-hot settings can share credit."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    sys.stdout.write(render(args.csv_path, args.top))


if __name__ == "__main__":
    main()
