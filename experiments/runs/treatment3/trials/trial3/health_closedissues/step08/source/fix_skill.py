#!/usr/bin/env python3
"""Treatment 3 / Trial 3 / Step 08 independent minimal-change evaluator."""
from __future__ import annotations
import argparse, csv, json, math, statistics, time
from pathlib import Path

REQUIRED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
CONFIG = REQUIRED[3:]
OUTCOMES = REQUIRED[:3]

def load_dataset(path: Path):
    rows, rejected = [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in REQUIRED):
            raise ValueError("missing required columns")
        for line, raw in enumerate(reader, 2):
            try:
                out = {c: float(raw[c]) for c in OUTCOMES}
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in out.values()):
                    raise ValueError("outcome outside [0,1]")
                cfg = {"N_estimators": int(raw["N_estimators"]),
                       "criterion": raw["criterion"].strip(),
                       "Min_sample_leaves": int(raw["Min_sample_leaves"]),
                       "Min_impurity_decrease": float(raw["Min_impurity_decrease"]),
                       "Max_depth": int(raw["Max_depth"])}
                if cfg["N_estimators"] <= 0 or cfg["Min_sample_leaves"] <= 0 or cfg["Max_depth"] <= 0 or cfg["Min_impurity_decrease"] < 0 or not cfg["criterion"]:
                    raise ValueError("invalid configuration")
                rows.append({**out, **cfg, "line": line})
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                rejected.append(f"line {line}: {exc}")
    if not rows:
        raise ValueError("no valid rows")
    return rows, rejected

def _median_baseline(rows):
    numeric = ["N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
    med = {c: statistics.median(float(r[c]) for r in rows) for c in numeric}
    counts = {}
    for r in rows: counts[r["criterion"]] = counts.get(r["criterion"], 0) + 1
    modal = max(counts, key=counts.get)
    def distance(r):
        return sum((float(r[c]) - med[c]) ** 2 for c in numeric) + (0 if r["criterion"] == modal else 100)
    return min(rows, key=distance), med, modal

def evaluate(rows):
    baseline, medians, modal = _median_baseline(rows)
    candidates = []
    for row in rows:
        changed = [c for c in CONFIG if row[c] != baseline[c]]
        if len(changed) == 1:
            error_gain = baseline["MRE-"] - row["MRE-"]
            quality_delta = (row["ACC+"] + row["PRED40+"]) - (baseline["ACC+"] + baseline["PRED40+"])
            candidates.append((error_gain, quality_delta, row, changed[0]))
    # Error is primary; quality is the tie-breaker, so quality is preserved as much as possible.
    improving = [x for x in candidates if x[0] > 0]
    best = max(improving, key=lambda x: (x[0], x[1])) if improving else None
    return baseline, best, len(candidates), medians, modal

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", type=Path); ap.add_argument("--json", action="store_true")
    args = ap.parse_args(); started = time.perf_counter()
    rows, rejected = load_dataset(args.dataset)
    baseline, best, candidate_count, medians, modal = evaluate(rows)
    result = {"dataset": str(args.dataset), "valid_rows": len(rows), "rejected_rows": len(rejected), "candidate_count": candidate_count,
              "baseline": {c: baseline[c] for c in REQUIRED}, "medians": medians, "modal_criterion": modal,
              "recommendation": None, "runtime_seconds": time.perf_counter() - started}
    if best:
        gain, qdelta, row, changed = best
        result["recommendation"] = {"changed_parameter": changed, "revised": {c: row[c] for c in REQUIRED},
                                    "error_gain": gain, "quality_delta_acc_plus_pred40_plus": qdelta,
                                    "baseline_error": baseline["MRE-"], "revised_error": row["MRE-"],
                                    "baseline_quality": [baseline["ACC+"], baseline["PRED40+"]],
                                    "revised_quality": [row["ACC+"], row["PRED40+"]], "revised_line": row["line"]}
    if args.json: print(json.dumps(result, sort_keys=True))
    else: print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
