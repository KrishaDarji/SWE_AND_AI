#!/usr/bin/env python3
"""Certify whether one categorical process group costs less than another."""
import argparse, csv, math, random, sys

def read_groups(path, feature, left, right, cost):
    groups = {left: [], right: []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            value = row[feature].strip()
            if value in groups:
                groups[value].append(float(row[cost]))
    if not groups[left] or not groups[right]:
        raise ValueError("both groups must contain at least one row")
    return groups

def mean(xs): return sum(xs) / len(xs)

def percentile(xs, p):
    ys = sorted(xs)
    if len(ys) == 1: return ys[0]
    index = (len(ys) - 1) * p
    lo, hi = int(index), math.ceil(index)
    return ys[lo] + (ys[hi] - ys[lo]) * (index - lo)

def certify(left, right, resamples=20000, seed=20261026):
    observed = mean(left) - mean(right)
    rng = random.Random(seed)
    pooled = left + right
    n_left = len(left)
    null_extreme = 0
    boot = []
    for _ in range(resamples):
        rng.shuffle(pooled)
        null_extreme += abs(mean(pooled[:n_left]) - mean(pooled[n_left:])) >= abs(observed)
        boot.append(mean([rng.choice(left) for _ in left]) - mean([rng.choice(right) for _ in right]))
    p_value = (null_extreme + 1) / (resamples + 1)
    low, high = percentile(boot, .025), percentile(boot, .975)
    return observed, p_value, low, high

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--feature", default="pmat")
    p.add_argument("--left", default="h")
    p.add_argument("--right", default="l")
    p.add_argument("--cost", default="Effort-")
    p.add_argument("--resamples", type=int, default=20000)
    args = p.parse_args(argv)
    groups = read_groups(args.csv_path, args.feature, args.left, args.right, args.cost)
    diff, p_value, low, high = certify(groups[args.left], groups[args.right], args.resamples)
    less = diff < 0 and high < 0 and p_value < .05
    verdict = "GENUINELY LESS COSTLY" if less else "NOT CERTIFIED AS LESS COSTLY"
    print("CERTIFY")
    print(f"feature={args.feature} left={args.left} right={args.right} cost={args.cost}")
    print(f"n_left={len(groups[args.left])} n_right={len(groups[args.right])}")
    print(f"mean_left={mean(groups[args.left]):.4f} mean_right={mean(groups[args.right]):.4f}")
    print(f"mean_difference_left_minus_right={diff:.4f}")
    print(f"permutation_p_value={p_value:.6f} bootstrap_95ci=[{low:.4f}, {high:.4f}]")
    print(f"verdict={verdict}")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (KeyError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(2)
