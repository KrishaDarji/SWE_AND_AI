#!/usr/bin/env python3
"""Choose a balanced configuration from the Health-ClosedIssues benchmark CSV."""
import argparse
import csv
import math
import os
import time

REQUIRED = ("MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
CRITERIA = {"absolute_error", "poisson", "squared_error"}
WEIGHTS = {"acc": 0.40, "pred40": 0.35, "mre": 0.25}


def _number(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError("non-finite " + name)
    return value


def load_candidates(path):
    """Load rows, returning (valid_rows, invalid_count, validation_notes)."""
    valid, invalid, notes = [], 0, []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(x not in reader.fieldnames for x in REQUIRED):
            raise ValueError("missing required columns")
        for line, row in enumerate(reader, 2):
            try:
                if any(row.get(c, "").strip() == "" for c in REQUIRED):
                    raise ValueError("missing value")
                metrics = {c: _number(row, c) for c in ("MRE-", "ACC+", "PRED40+")}
                params = {c: _number(row, c) for c in REQUIRED[3:] if c != "criterion"}
                if row["criterion"] not in CRITERIA:
                    raise ValueError("unsupported criterion")
                if not (0 <= metrics["MRE-"] <= 1 and 0 <= metrics["ACC+"] <= 1 and
                        0 <= metrics["PRED40+"] <= 1):
                    raise ValueError("objective outside [0,1]")
                if params["N_estimators"] < 1 or params["Min_sample_leaves"] < 1 or params["Max_depth"] < 1 or params["Min_impurity_decrease"] < 0:
                    raise ValueError("infeasible hyperparameter")
                valid.append(row)
            except (KeyError, TypeError, ValueError) as exc:
                invalid += 1
                if len(notes) < 5:
                    notes.append("line %d: %s" % (line, exc))
    if not valid:
        raise ValueError("no feasible candidates")
    return valid, invalid, notes


def choose(rows):
    """Maximize ACC and PRED40, minimize MRE, with stated weights."""
    def vals(key): return [float(r[key]) for r in rows]
    mins = {k: min(vals(k)) for k in ("MRE-", "ACC+", "PRED40+")}
    maxs = {k: max(vals(k)) for k in ("MRE-", "ACC+", "PRED40+")}
    def norm(row, key, higher):
        span = maxs[key] - mins[key]
        if span == 0: return 1.0
        x = (float(row[key]) - mins[key]) / span
        return x if higher else 1.0 - x
    scored = []
    for row in rows:
        score = (WEIGHTS["acc"] * norm(row, "ACC+", True) +
                 WEIGHTS["pred40"] * norm(row, "PRED40+", True) +
                 WEIGHTS["mre"] * norm(row, "MRE-", False))
        scored.append((score, row))
    return max(scored, key=lambda pair: (pair[0], float(pair[1]["ACC+"]), float(pair[1]["PRED40+"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    started = time.perf_counter()
    rows, invalid, notes = load_candidates(args.dataset)
    score, row = choose(rows)
    elapsed = time.perf_counter() - started
    print("dataset=%s" % os.path.basename(args.dataset))
    print("objectives=ACC+ maximize (0.40), PRED40+ maximize (0.35), MRE- minimize (0.25)")
    print("candidates_valid=%d candidates_invalid=%d" % (len(rows), invalid))
    print("selected_score=%.9f" % score)
    print("selected_configuration=" + ",".join("%s=%s" % (k, row[k]) for k in REQUIRED[3:]))
    print("metrics=MRE-%.9f,ACC+%.9f,PRED40+%.9f" % (float(row["MRE-"]), float(row["ACC+"]), float(row["PRED40+"])))
    print("runtime_seconds=%.6f" % elapsed)
    if notes: print("validation_notes=" + " | ".join(notes))


if __name__ == "__main__":
    main()
