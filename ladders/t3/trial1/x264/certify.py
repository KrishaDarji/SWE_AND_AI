"""Statistical certification of a performance improvement in X264 measurements."""
import csv
import math
import random
import sys
import time
from sklearn.utils.validation import check_array


def read_measurements(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    target = list(rows[0].keys())[-1]
    values = [float(row[target].strip()) for row in rows]
    return rows, target, values


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values):
    return sum(values) / len(values)


def certify(values, seed=20261029, repetitions=10000):
    """Return effect, 95% bootstrap CI, and one-sided permutation p-value.

    Lower target values are treated as improvements. The candidate group is
    the lowest-performing decile, while the reference group is all others.
    """
    values = [float(x) for x in check_array(values, ensure_2d=False, dtype="numeric")]
    if len(values) < 10:
        raise ValueError("at least 10 measurements are required")
    cutoff = percentile(values, 0.10)
    candidate = [v for v in values if v <= cutoff]
    reference = [v for v in values if v > cutoff]
    observed = mean(reference) - mean(candidate)
    rng = random.Random(seed)
    boot = []
    for _ in range(repetitions):
        a = [candidate[rng.randrange(len(candidate))] for _ in candidate]
        b = [reference[rng.randrange(len(reference))] for _ in reference]
        boot.append(mean(b) - mean(a))
    boot.sort()
    lo, hi = percentile(boot, 0.025), percentile(boot, 0.975)
    combined = candidate + reference
    count = 0
    for _ in range(repetitions):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        simulated = mean(shuffled[len(candidate):]) - mean(shuffled[:len(candidate)])
        if simulated >= observed:
            count += 1
    p_value = (count + 1) / (repetitions + 1)
    credible = observed > 0 and lo > 0 and p_value < 0.05
    return {
        "n": len(values), "candidate_n": len(candidate), "reference_n": len(reference),
        "cutoff": cutoff, "candidate_mean": mean(candidate),
        "reference_mean": mean(reference), "improvement": observed,
        "ci95_low": lo, "ci95_high": hi, "p_value": p_value,
        "credible": credible,
    }


def main(dataset_path):
    started = time.perf_counter()
    _, target, values = read_measurements(dataset_path)
    result = certify(values)
    result["target"] = target
    result["runtime_seconds"] = time.perf_counter() - started
    print("target:", result["target"])
    print("measurements:", result["n"], "candidate:", result["candidate_n"], "reference:", result["reference_n"])
    print("improvement (reference mean - candidate mean): %.6f" % result["improvement"])
    print("95%% bootstrap CI: [%.6f, %.6f]" % (result["ci95_low"], result["ci95_high"]))
    print("one-sided permutation p-value: %.6f" % result["p_value"])
    print("statistically credible:", result["credible"])
    print("runtime_seconds: %.6f" % result["runtime_seconds"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python certify.py DATASET.csv")
    main(sys.argv[1])
