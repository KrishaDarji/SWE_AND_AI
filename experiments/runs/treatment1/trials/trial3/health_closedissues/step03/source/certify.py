#!/usr/bin/env python3
"""Certify whether a configuration family reliably outperforms the others."""
import argparse, csv, random, statistics

REQUIRED = ("MRE-", "ACC+", "PRED40+", "criterion")

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(k not in rows[0] for k in REQUIRED):
        raise ValueError("CSV is missing required columns")
    return rows

def score(row):
    # Equal-weight direction-aware utility; each metric is already bounded or comparable.
    return float(row["ACC+"]) + float(row["PRED40+"]) - float(row["MRE-"])

def bootstrap(values, rng, samples=4000):
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    means.sort()
    return means[int(.025 * samples)], means[int(.975 * samples)]

def certify(path, seed=1729, samples=4000):
    rows = load(path)
    groups = {}
    for row in rows:
        groups.setdefault(row["criterion"], []).append(score(row))
    rng = random.Random(seed)
    stats = {g: (len(v), statistics.fmean(v), bootstrap(v, rng, samples)) for g, v in groups.items()}
    winner = max(stats, key=lambda g: stats[g][1])
    # Reliable means: winner's lower CI exceeds every rival's point estimate.
    reliable = all(stats[winner][2][0] > stats[g][1] for g in stats if g != winner)
    print("Configuration family: criterion")
    for g in sorted(stats):
        n, mean, (lo, hi) = stats[g]
        print(f"{g}: n={n}, mean_utility={mean:.6f}, 95%_bootstrap_CI=[{lo:.6f}, {hi:.6f}]")
    if reliable:
        print(f"CERTIFIED: {winner} is reliably better than every other family.")
    else:
        print("NOT CERTIFIED: no family is reliably better under the stated rule.")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--samples", type=int, default=4000)
    args = ap.parse_args()
    if args.samples < 100:
        ap.error("--samples must be at least 100")
    return certify(args.csv_path, args.seed, args.samples)

if __name__ == "__main__":
    main()
