import csv
import math
import random
import statistics
from pathlib import Path

from sklearn.linear_model import LinearRegression


DATASET = Path("moot/optimize/config/X264_AllMeasurements.csv")
OUT = Path(__file__).resolve().parents[1]
CONFIG = [
    "no_asm", "no_8x8dct", "no_cabac", "no_deblock", "no_fast_pskip",
    "no_mbtree", "no_mixed_refs", "no_weightb", "rc_lookahead",
    "rc_lookahead_20", "rc_lookahead_40", "rc_lookahead_60", "ref",
    "ref_1", "ref_5", "ref_9",
]
TARGET = "Performance-"


def load_validate(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != CONFIG + [TARGET]:
        raise ValueError("unexpected header or empty dataset")
    parsed = []
    for number, row in enumerate(rows, 2):
        try:
            bits = [int(row[col].strip()) for col in CONFIG]
            score = float(row[TARGET].strip())
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid row %d: %s" % (number, exc))
        if any(bit not in (0, 1) for bit in bits) or not math.isfinite(score):
            raise ValueError("non-binary or non-finite value at row %d" % number)
        parsed.append((bits, score))
    if len({tuple(x[0]) for x in parsed}) != len(parsed):
        raise ValueError("duplicate configurations")
    return parsed


def bootstrap_mean(values, seed=607):
    rng = random.Random(seed)
    means = []
    for _ in range(2000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    return statistics.mean(values), means[40], means[1960]


def analyze(path=DATASET):
    data = load_validate(path)
    best_bits, best_score = max(data, key=lambda pair: pair[1])
    baseline = [score for bits, score in data if not any(bits)]
    # The full factorial has one observation per configuration; regression is
    # descriptive only and is not used to replace the observed optimum.
    model = LinearRegression().fit([bits for bits, score in data], [score for bits, score in data])
    effects = sorted(zip(CONFIG, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    mean, low, high = bootstrap_mean([score for bits, score in data])
    return data, best_bits, best_score, baseline, effects, (mean, low, high)


def render(path=DATASET):
    data, best, score, baseline, effects, interval = analyze(path)
    names = [name for name, bit in zip(CONFIG, best) if bit]
    base = baseline[0] if baseline else None
    uplift = (score - base) / base * 100 if base else None
    lines = [
        "x264 configuration recommendation — Treatment 3 / Trial 3 / Dataset 1",
        "",
        "Recommendation: use the measured configuration listed below (the observed maximum).",
        "Enabled indicators: " + ", ".join(names),
        "",
        "Evidence:",
        "- Validated %d unique rows; observed Performance- range %.3f to %.3f (larger is treated as better)." % (len(data), min(s for b, s in data), max(s for b, s in data)),
        "- Recommended measured Performance-: %.3f." % score,
        "- The all-zero reference is %.3f; the observed difference is %.1f%%." % (base, uplift) if base else "- No all-zero reference row was present; no baseline uplift is claimed.",
        "- Bootstrap 95%% interval for the dataset-wide mean Performance-: %.3f to %.3f (mean %.3f; resampling uncertainty only)." % (interval[1], interval[2], interval[0]),
        "- Largest absolute descriptive linear coefficients: " + "; ".join("%s=%+.3f" % x for x in effects[:4]) + ".",
        "",
        "Trade-offs and alternatives:",
        "- Enabling CABAC, deblocking, mb-tree, weighted prediction, references, and lookahead can improve compression efficiency/visual quality, but may increase CPU time, latency, or memory; the CSV measures none of those costs or quality metrics.",
        "- A lower-cost alternative should be selected from a separately measured CPU/latency/quality Pareto set; this dataset alone cannot rank such an alternative.",
        "",
        "Caveats for the administrator:",
        "- This is a historical association in one benchmark. The full-factorial rows support comparison within this dataset, but do not establish that any switch caused the difference in production.",
        "- Assumes higher Performance- is desirable, the benchmark is representative, and configuration validity transfers to the target encoder/build.",
        "- Validate the recommendation with repeated production-like encodes, CPU time, memory, bitrate/file size, and an objective or subjective quality measure before rollout; use a canary if cost or quality risk is material.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render(), end="")
