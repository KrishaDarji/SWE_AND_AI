"""Independent, reproducible anomaly scan for Health-ClosedIssues results."""
import csv
import math
import os
from statistics import median


EXPECTED = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
METRICS = ["MRE-", "ACC+", "PRED40+"]
CONFIG = ["N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth"]


def _mad_score(value, values):
    med = median(values)
    mad = median([abs(x - med) for x in values])
    scale = 1.4826 * mad
    return abs(value - med) / scale if scale else (0.0 if value == med else float("inf"))


def load_validate(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != EXPECTED:
        raise ValueError("unexpected or empty header")
    parsed = []
    for i, row in enumerate(rows):
        try:
            out = {k: row[k] for k in EXPECTED}
            for k in EXPECTED:
                if k != "criterion":
                    out[k] = float(row[k])
            if any(not math.isfinite(out[k]) for k in EXPECTED if k != "criterion"):
                raise ValueError("non-finite")
            if any(not 0 <= out[k] <= 1 for k in METRICS):
                raise ValueError("metric outside [0,1]")
            if out["N_estimators"] < 1 or out["Min_sample_leaves"] < 1 or out["Max_depth"] < 1:
                raise ValueError("invalid positive hyperparameter")
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid row {i}: {exc}") from exc
        out["row"] = i + 2
        parsed.append(out)
    return parsed


def detect(rows, seed=20261029):
    fields = METRICS + [k for k in CONFIG if k != "criterion"]
    stats = {}
    for k in fields:
        values = [x[k] for x in rows]
        med = median(values)
        mad = median([abs(x - med) for x in values])
        stats[k] = (med, 1.4826 * mad)
    robust = [sum(_mad_score(r[k], [stats[k][0], stats[k][0] + stats[k][1]])
                  for k in fields) for r in rows]
    # A scale-free robust median/MAD ensemble is deterministic.
    def rank(values, value):
        return sum(v <= value for v in values) / len(values)
    combined = [rank(robust, robust[i]) for i in range(len(rows))]
    best = max(range(len(rows)), key=lambda i: (combined[i], i))
    r = rows[best]
    return {"row": r["row"], "configuration": {k: r[k] for k in CONFIG},
            "metrics": {k: r[k] for k in METRICS}, "isolation_score": None,
            "robust_score": robust[best], "anomaly_score": combined[best],
            "metric_mad_scores": {k: _mad_score(r[k], [stats[k][0], stats[k][0] + stats[k][1]]) for k in METRICS},
            "n_rows": len(rows), "seed": seed}


def main(path):
    rows = load_validate(path)
    result = detect(rows)
    print(f"validated_rows={len(rows)}")
    print(f"suspicious_csv_row={result['row']} anomaly_score={result['anomaly_score']:.6f}")
    print("configuration=" + repr(result["configuration"]))
    print("metrics=" + repr(result["metrics"]))
    print("metric_mad_scores=" + repr({k: round(v, 4) for k, v in result["metric_mad_scores"].items()}))
    print("method=robust median/MAD ensemble rank; seed=20261029")
    return result


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "moot/optimize/hpo/Health-ClosedIssues0000.csv")
