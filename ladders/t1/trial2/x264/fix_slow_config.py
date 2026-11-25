#!/usr/bin/env python3
"""Recommend one setting flip that improves a slow x264 configuration."""
import argparse
import csv
import json
from pathlib import Path


def read_measurements(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no measurements")
    performance = "Performance-"
    for row in rows:
        row[performance] = float(row[performance].strip())
    return rows


def setting_columns(rows):
    return [name for name in rows[0] if name != "Performance-"]


def suggest(rows, config=None):
    columns = setting_columns(rows)
    if config is None:
        current = min(rows, key=lambda row: row["Performance-"])
    else:
        missing = set(columns) - set(config)
        if missing:
            raise ValueError("config is missing: " + ", ".join(sorted(missing)))
        current = min(rows, key=lambda row: sum(str(config[c]) != row[c] for c in columns))
    candidates = []
    for row in rows:
        distance = sum(row[c] != current[c] for c in columns)
        if distance == 1 and row["Performance-"] > current["Performance-"]:
            changed = next(c for c in columns if row[c] != current[c])
            candidates.append((row["Performance-"] - current["Performance-"], changed, row))
    if not candidates:
        return {"status": "no_improvement", "current_performance": current["Performance-"], "setting": None}
    gain, changed, target = max(candidates, key=lambda item: (item[0], item[1]))
    return {"status": "improvement", "setting": changed, "from": current[changed],
            "to": target[changed], "current_performance": current["Performance-"],
            "expected_performance": target["Performance-"], "gain": gain}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, help="JSON object containing setting values")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    print(json.dumps(suggest(read_measurements(args.dataset), config), sort_keys=True))


if __name__ == "__main__":
    main()
