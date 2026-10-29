#!/usr/bin/env python3
"""Independent justification analysis for Health-ClosedIssues0000.csv."""
import argparse, csv, math, random, statistics, time
from pathlib import Path

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

def load_dataset(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = set(OUTCOMES + CONFIG)
    if not rows or not required.issubset(rows[0]):
        raise ValueError("missing required columns or empty dataset")
    parsed = []
    for i, row in enumerate(rows, 2):
        try:
            out = {k: float(row[k]) for k in OUTCOMES}
            cfg = {"N_estimators": int(row["N_estimators"]), "criterion": row["criterion"],
                   "Min_sample_leaves": int(row["Min_sample_leaves"]),
                   "Min_impurity_decrease": float(row["Min_impurity_decrease"]),
                   "Max_depth": int(row["Max_depth"])}
        except (ValueError, TypeError) as e:
            raise ValueError(f"invalid row {i}: {e}")
        if cfg["N_estimators"] <= 0 or cfg["Min_sample_leaves"] <= 0 or cfg["Max_depth"] <= 0:
            raise ValueError(f"non-positive configuration at row {i}")
        if not all(math.isfinite(v) for v in out.values()):
            raise ValueError(f"non-finite outcome at row {i}")
        if not (0 <= out["MRE-"] <= 1 and 0 <= out["ACC+"] <= 1 and 0 <= out["PRED40+"] <= 1):
            raise ValueError(f"outcome outside [0,1] at row {i}")
        parsed.append({**cfg, **out})
    return parsed

def utility(row):
    # Equal emphasis on prediction quality, with MRE converted to a higher-is-better score.
    return (1.0 - row["MRE-"] + row["ACC+"] + row["PRED40+"]) / 3.0

def recommend(rows, seed=20261029, bootstraps=2000):
    ranked = sorted(rows, key=utility, reverse=True)
    best = ranked[0]
    rng = random.Random(seed)
    values = []
    n = len(rows)
    for _ in range(bootstraps):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        values.append(max(map(utility, sample)))
    values.sort()
    lo, hi = values[int(.025 * bootstraps)], values[int(.975 * bootstraps) - 1]
    mean = statistics.mean(values)
    return best, utility(best), (lo, hi, mean), ranked[:5]

def write_metrics(path, best, score, interval):
    fields = ["configuration", "utility", "utility_ci95_low", "utility_ci95_high", "utility_bootstrap_mean", *OUTCOMES]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerow({"configuration": ";".join(f"{k}={best[k]}" for k in CONFIG), "utility": f"{score:.6f}",
                    "utility_ci95_low": f"{interval[0]:.6f}", "utility_ci95_high": f"{interval[1]:.6f}",
                    "utility_bootstrap_mean": f"{interval[2]:.6f}", **{k: f"{best[k]:.6f}" for k in OUTCOMES}})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(); start = time.perf_counter()
    rows = load_dataset(args.dataset); best, score, interval, top = recommend(rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_metrics(out / "analysis_metrics.csv", best, score, interval)
    print(f"rows={len(rows)} validated=true")
    print("recommended=" + ",".join(f"{k}={best[k]}" for k in CONFIG))
    print(f"outcomes=MRE-:{best['MRE-']:.6f},ACC+:{best['ACC+']:.6f},PRED40+:{best['PRED40+']:.6f}")
    print(f"utility={score:.6f} bootstrap95={interval[0]:.6f}..{interval[1]:.6f} bootstrap_mean={interval[2]:.6f}")
    print("alternatives=" + " | ".join(f"{utility(r):.6f}:" + ",".join(str(r[k]) for k in CONFIG) for r in top[1:4]))
    print(f"runtime_seconds={time.perf_counter()-start:.6f}")

if __name__ == "__main__": main()
