#!/usr/bin/env python3
"""Find the smallest measured x264 setting change that improves a slow row."""
import argparse
import csv
import os
from dataclasses import dataclass

GROUPS = (("rc_lookahead", "rc_lookahead_20", "rc_lookahead_40", "rc_lookahead_60"),
          ("ref", "ref_1", "ref_5", "ref_9"))
RESULT = "Performance-"

@dataclass(frozen=True)
class Recommendation:
    baseline: dict
    candidate: dict
    baseline_result: float
    revised_result: float

def load_dataset(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or RESULT not in reader.fieldnames:
            raise ValueError("missing Performance- result column")
        settings = [x for x in reader.fieldnames if x != RESULT]
        rows = list(reader)
    if not rows or not settings:
        raise ValueError("dataset must contain rows and settings")
    for row_number, row in enumerate(rows, 2):
        for name in settings:
            if row[name].strip() not in {"0", "1"}:
                raise ValueError("row %d: %s is not binary" % (row_number, name))
        try:
            row[RESULT] = float(row[RESULT].strip())
        except ValueError as exc:
            raise ValueError("row %d: invalid result" % row_number) from exc
    for group in GROUPS:
        if not set(group).issubset(settings):
            raise ValueError("incomplete one-hot group: %s" % (group,))
        for row in rows:
            # The base flag is always on; exactly one named alternative is on.
            if row[group[0]] != "1" or sum(int(row[x]) for x in group[1:]) != 1:
                raise ValueError("row violates one-hot group %s" % (group,))
    return settings, rows

def key(row, settings):
    return tuple(row[x] for x in settings)

def find_recommendation(settings, rows):
    # Under the documented assumption that larger Performance- means slower.
    slow = max(rows, key=lambda row: row[RESULT])
    baseline_result = slow[RESULT]
    baseline_key = key(slow, settings)
    candidates = []
    for row in rows:
        distance = sum(a != b for a, b in zip(baseline_key, key(row, settings)))
        if distance == 1 and row[RESULT] < baseline_result:
            changed = [x for x in settings if slow[x] != row[x]]
            candidates.append((row[RESULT], changed[0], row))
    if not candidates:
        return None, slow, None
    revised_result, _, candidate = min(candidates, key=lambda item: item[0])
    return Recommendation(slow, candidate, baseline_result, revised_result), slow, len(candidates)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    settings, rows = load_dataset(args.dataset)
    recommendation, baseline, candidate_count = find_recommendation(settings, rows)
    print("rows=%d settings=%d" % (len(rows), len(settings)))
    print("assumption=larger Performance- is slower; improvement means lower result")
    print("baseline_result=%.3f" % baseline[RESULT])
    print("baseline_settings=" + ",".join("%s=%s" % (x, baseline[x]) for x in settings))
    if recommendation is None:
        print("recommendation=none")
        print("reason=no measured one-setting change improves the slowest baseline")
        return 0
    changed = next(x for x in settings if recommendation.baseline[x] != recommendation.candidate[x])
    gain = recommendation.baseline_result - recommendation.revised_result
    print("candidate_count=%d" % candidate_count)
    print("changed_setting=%s (%s -> %s)" % (changed, recommendation.baseline[changed], recommendation.candidate[changed]))
    print("revised_result=%.3f" % recommendation.revised_result)
    print("gain=%.3f (%.2f%% lower)" % (gain, 100 * gain / recommendation.baseline_result))
    print("revised_settings=" + ",".join("%s=%s" % (x, recommendation.candidate[x]) for x in settings))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
