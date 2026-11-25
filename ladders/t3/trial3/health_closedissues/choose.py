#!/usr/bin/env python3
"""Treatment 3 / Trial 3 / Step 07 configuration chooser.

The input is a measured configuration table.  Selection is deliberately
independent of other experiment steps: rows are validated, scored, and ranked
using only the columns in this file.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REQUIRED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
CONFIG = ["N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth"]


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def load_rows(path: Path) -> Tuple[List[Dict[str, object]], int, List[str]]:
    rows, rejected, reasons = [], 0, []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in REQUIRED):
            raise ValueError("missing required columns")
        for line, raw in enumerate(reader, 2):
            try:
                objective = {c: _finite(raw[c]) for c in REQUIRED[:3]}
                if not (0 <= objective["MRE-"] <= 1 and 0 <= objective["ACC+"] <= 1
                        and 0 <= objective["PRED40+"] <= 1):
                    raise ValueError("objective outside [0,1]")
                config = {
                    "N_estimators": int(raw["N_estimators"]),
                    "criterion": raw["criterion"].strip(),
                    "Min_sample_leaves": int(raw["Min_sample_leaves"]),
                    "Min_impurity_decrease": _finite(raw["Min_impurity_decrease"]),
                    "Max_depth": int(raw["Max_depth"]),
                }
                if config["N_estimators"] <= 0 or config["Min_sample_leaves"] <= 0 \
                        or config["Max_depth"] <= 0 or config["Min_impurity_decrease"] < 0 \
                        or not config["criterion"]:
                    raise ValueError("infeasible hyperparameter")
                rows.append({**objective, **config, "line": line})
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                rejected += 1
                if len(reasons) < 5:
                    reasons.append(f"line {line}: {exc}")
    if not rows:
        raise ValueError("no valid configurations")
    return rows, rejected, reasons


def choose(rows: Iterable[Dict[str, object]]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rows = list(rows)
    # Min-max normalization makes the three measured objectives comparable.
    def span(name: str) -> Tuple[float, float]:
        values = [float(r[name]) for r in rows]
        return min(values), max(values)
    bounds = {name: span(name) for name in ("MRE-", "ACC+", "PRED40+")}
    scored = []
    for row in rows:
        def norm(name: str, value: float, maximize: bool) -> float:
            lo, hi = bounds[name]
            result = 0.5 if hi == lo else (value - lo) / (hi - lo)
            return result if maximize else 1.0 - result
        accuracy = norm("ACC+", float(row["ACC+"]), True)
        quality = (norm("PRED40+", float(row["PRED40+"]), True)
                   + norm("MRE-", float(row["MRE-"]), False)) / 2.0
        score = 0.5 * accuracy + 0.5 * quality
        scored.append({**row, "accuracy_score": accuracy,
                       "prediction_quality_score": quality, "composite_score": score})
    scored.sort(key=lambda r: (-float(r["composite_score"]), float(r["MRE-"]),
                               -float(r["ACC+"]), -float(r["PRED40+"]), int(r["line"])))
    return scored[0], scored


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        rows, rejected, reasons = load_rows(args.dataset)
        best, ranked = choose(rows)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"valid_rows={len(rows)} rejected_rows={rejected}")
    for reason in reasons:
        print(f"rejection_example={reason}")
    print("selected_configuration=" + ",".join(f"{k}={best[k]}" for k in CONFIG))
    print(f"MRE-={best['MRE-']:.9f} ACC+={best['ACC+']:.9f} PRED40+={best['PRED40+']:.9f}")
    print(f"accuracy_score={best['accuracy_score']:.9f} prediction_quality_score={best['prediction_quality_score']:.9f} composite_score={best['composite_score']:.9f}")
    print(f"runner_up_composite_score={ranked[1]['composite_score']:.9f}" if len(ranked) > 1 else "runner_up_composite_score=NA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
