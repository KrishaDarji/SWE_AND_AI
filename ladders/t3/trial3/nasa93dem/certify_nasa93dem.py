"""Independent certification analysis for NASA93DEM."""
import argparse, csv, math, random, statistics
from pathlib import Path

TARGET = "Effort-"
GROUP = "centerX"
SEED = 303
PERMUTATIONS = 10000
BOOTSTRAPS = 10000

def mean(xs): return statistics.fmean(xs)

def percentile(xs, q):
    ys = sorted(xs); pos = (len(ys)-1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi]-ys[lo])*(pos-lo)

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or TARGET not in rows[0] or GROUP not in rows[0]:
        raise ValueError("required target/group columns are missing")
    values = []
    for i, row in enumerate(rows, 2):
        try: effort = float(row[TARGET].strip()); group = row[GROUP].strip()
        except (TypeError, ValueError) as e: raise ValueError(f"invalid row {i}") from e
        if not group or not math.isfinite(effort): raise ValueError(f"invalid row {i}")
        values.append((group, effort))
    return values

def certify(path):
    values = load(path)
    groups = {}
    for group, effort in values: groups.setdefault(group, []).append(effort)
    eligible = sorted(groups, key=lambda g: (-len(groups[g]), g))
    if len(eligible) < 2: raise ValueError("fewer than two process groups")
    a, b = eligible[:2]
    # a is the larger group; orient contrast as higher mean minus lower mean.
    ma, mb = mean(groups[a]), mean(groups[b])
    low, high = (a, b) if ma <= mb else (b, a)
    x, y = groups[low], groups[high]
    observed = mean(y) - mean(x)
    pooled = math.sqrt(((len(x)-1)*statistics.variance(x) + (len(y)-1)*statistics.variance(y)) /
                       (len(x)+len(y)-2))
    effect = observed / pooled if pooled else float("inf")
    rng = random.Random(SEED)
    ci_samples = []
    for _ in range(BOOTSTRAPS):
        bx = [x[rng.randrange(len(x))] for _ in x]
        by = [y[rng.randrange(len(y))] for _ in y]
        ci_samples.append(mean(by)-mean(bx))
    ci = (percentile(ci_samples, .025), percentile(ci_samples, .975))
    combined = x + y; n_x = len(x); extreme = 0
    for _ in range(PERMUTATIONS):
        shuffled = combined[:]; rng.shuffle(shuffled)
        contrast = mean(shuffled[n_x:]) - mean(shuffled[:n_x])
        if contrast >= observed: extreme += 1
    p = (extreme + 1) / (PERMUTATIONS + 1)
    credible = p < .05 and ci[0] > 0
    return {"n_total": len(values), "groups": len(groups), "low": low, "high": high,
            "n_low": len(x), "n_high": len(y), "low_mean": mean(x), "high_mean": mean(y),
            "effect": effect, "difference": observed, "ci_low": ci[0], "ci_high": ci[1],
            "p": p, "credible": credible}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", type=Path); args = ap.parse_args()
    r = certify(args.dataset)
    print(f"validated_rows={r['n_total']} process_groups={r['groups']} target={TARGET} feature={GROUP}")
    print(f"lower_effort_group={r['low']} n={r['n_low']} mean={r['low_mean']:.6f}")
    print(f"higher_effort_group={r['high']} n={r['n_high']} mean={r['high_mean']:.6f}")
    print(f"mean_difference_high_minus_low={r['difference']:.6f} standardized_effect={r['effect']:.6f}")
    print(f"bootstrap_95ci=[{r['ci_low']:.6f}, {r['ci_high']:.6f}] permutations={PERMUTATIONS} seed={SEED}")
    print(f"one_sided_permutation_p={r['p']:.6f} statistically_credible={r['credible']}")
    print("interpretation=association only; this observational comparison does not prove causation")

if __name__ == "__main__": main()
