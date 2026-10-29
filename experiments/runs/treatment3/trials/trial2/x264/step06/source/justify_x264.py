#!/usr/bin/env python3
"""Evidence-based, explicitly non-causal justification of an x264 setting."""
import argparse, csv, math, random
from pathlib import Path
from statistics import mean, stdev
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

RESPONSE = "Performance-"

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or RESPONSE not in rows[0]: raise ValueError("missing rows or response")
    features = [k for k in rows[0] if k != RESPONSE]
    if any(not k for k in features): raise ValueError("empty feature name")
    X, y = [], []
    for row in rows:
        try:
            vals = [float(row[k]) for k in features]; target = float(row[RESPONSE])
        except (KeyError, TypeError, ValueError) as e: raise ValueError("non-numeric data") from e
        if any(v not in (0.0, 1.0) for v in vals) or not math.isfinite(target):
            raise ValueError("features must be binary and response finite")
        X.append(vals); y.append(target)
    if len(set(tuple(x) for x in X)) != len(X): raise ValueError("duplicate configurations")
    return features, X, y, rows

def bootstrap_mean(values, n=4000, seed=17):
    rng = random.Random(seed); samples=[]; m=len(values)
    for _ in range(n): samples.append(mean(values[rng.randrange(m)] for _ in range(m)))
    samples.sort(); return samples[int(.025*n)], samples[int(.975*n)]

def justify(path):
    features, X, y, rows = load(path)
    best_i = max(range(len(y)), key=y.__getitem__); best=X[best_i]
    ranked=sorted(y, reverse=True); top10=ranked[:10]
    # Do not standardize: constant columns are valid controls in this factorial table.
    variable=[j for j in range(len(features)) if len({row[j] for row in X}) > 1]
    Xm=[[row[j] for j in variable] for row in X]
    model=Ridge(alpha=1.0)
    cv=cross_val_score(model, Xm, y, cv=KFold(5, shuffle=True, random_state=17), scoring="r2")
    model.fit(Xm,y); pred=model.predict([[best[j] for j in variable]])[0]
    residual=[a-b for a,b in zip(y, model.predict(Xm))]
    residual_boot=bootstrap_mean(residual, seed=29)
    all_ci=bootstrap_mean(y, seed=31)
    baseline=mean(y); best_gain=100*(y[best_i]/baseline-1)
    cfg=dict(zip(features, [int(v) for v in best]))
    print("X264 JUSTIFICATION")
    print(f"rows={len(y)} features={len(features)} response={RESPONSE}")
    print("recommended_configuration=" + ",".join(f"{k}={v}" for k,v in cfg.items()))
    print(f"observed_performance={y[best_i]:.3f}; dataset_mean={baseline:.3f}; gain_vs_mean_pct={best_gain:.2f}")
    print(f"top10_mean={mean(top10):.3f}; top10_min={min(top10):.3f}; top10_max={max(top10):.3f}")
    print(f"bootstrap_95ci_dataset_mean={all_ci[0]:.3f},{all_ci[1]:.3f}")
    print(f"model_prediction_at_recommendation={pred:.3f}; residual_bootstrap_95ci={pred+residual_boot[0]:.3f},{pred+residual_boot[1]:.3f}")
    print(f"five_fold_r2_mean={mean(cv):.4f}; five_fold_r2_sd={stdev(cv):.4f}")
    print("decision=Recommend the observed maximum for throughput-sensitive deployment, subject to a quality gate.")
    print("tradeoff=The file measures performance only; it does not measure bitrate, visual quality, latency variance, power, or compatibility.")
    print("alternatives=Choose a top-10 configuration if quality/robustness testing rejects the maximum; validate CABAC, deblock, mbtree, and reference/lookahead settings on representative clips.")
    print("causal_boundary=These are historical associations in this dataset. The factorial design supports controlled contrasts, but this analysis alone does not establish real-world causal effects or quality effects.")
    return 0

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("dataset", type=Path); args=ap.parse_args()
    try: raise SystemExit(justify(args.dataset))
    except (OSError, ValueError) as e: print(f"ERROR: {e}"); raise SystemExit(2)
