#!/usr/bin/env python3
"""Explain which hyperparameter settings accompany high error or low accuracy."""
import argparse, csv, math, statistics

TARGETS = ("MRE-", "ACC+", "PRED40+")
SETTINGS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("dataset contains no data rows")
    missing = set(TARGETS + SETTINGS) - set(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    return rows

def _numbers(rows, field):
    return [float(row[field]) for row in rows]

def _correlation(left, right):
    mean_left, mean_right = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0

def _median_split(values, outcomes):
    midpoint = statistics.median(outcomes)
    high = [value for value, outcome in zip(values, outcomes) if outcome >= midpoint]
    low = [value for value, outcome in zip(values, outcomes) if outcome < midpoint]
    return (statistics.mean(high) if high else 0.0, statistics.mean(low) if low else 0.0, midpoint)

def _numeric_finding(rows, setting, error, accuracy):
    values = _numbers(rows, setting)
    error_r, accuracy_r = _correlation(values, error), _correlation(values, accuracy)
    high_error, low_error, error_cut = _median_split(values, error)
    low_accuracy, high_accuracy, accuracy_cut = _median_split(values, accuracy)
    score = (error_r - accuracy_r) / 2.0
    direction = "higher values" if score > 0 else "lower values"
    return (abs(score), f"{setting}: {direction} are associated with worse results "
            f"(MRE r={error_r:.3f}, ACC r={accuracy_r:.3f}; "
            f"mean setting in high-error half={high_error:.3f}, low-error half={low_error:.3f}; "
            f"low-accuracy half={low_accuracy:.3f}, high-accuracy half={high_accuracy:.3f}; "
            f"cuts MRE={error_cut:.3f}, ACC={accuracy_cut:.3f})")

def _categorical_finding(rows, setting, error, accuracy):
    groups = {}
    for row, error_value, accuracy_value in zip(rows, error, accuracy):
        groups.setdefault(row[setting], [[], []])
        groups[row[setting]][0].append(error_value)
        groups[row[setting]][1].append(accuracy_value)
    summaries = [f"{value}: MRE={statistics.mean(pair[0]):.3f}, ACC={statistics.mean(pair[1]):.3f}, n={len(pair[0])}"
                 for value, pair in sorted(groups.items())]
    error_means = [statistics.mean(pair[0]) for pair in groups.values()]
    accuracy_means = [statistics.mean(pair[1]) for pair in groups.values()]
    spread = (max(error_means) - min(error_means) if error_means else 0.0)
    spread += (max(accuracy_means) - min(accuracy_means) if accuracy_means else 0.0)
    return spread, f"{setting}: " + "; ".join(summaries)

def analyze(rows):
    error, accuracy = _numbers(rows, "MRE-"), _numbers(rows, "ACC+")
    findings = [_numeric_finding(rows, setting, error, accuracy)
                for setting in SETTINGS if setting != "criterion"]
    findings.append(_categorical_finding(rows, "criterion", error, accuracy))
    findings.sort(key=lambda item: item[0], reverse=True)
    lines = ["BLAME ANALYSIS", f"rows: {len(rows)}",
             "requirement: explain settings associated with high MRE or low ACC",
             "ranked findings:"]
    lines.extend(f"{index}. {finding}" for index, (_, finding) in enumerate(findings, 1))
    lines.append("interpretation: positive MRE association means higher error; negative ACC association means lower accuracy.")
    lines.append("limitation: these are observational associations over tested configurations, not causal effects or a controlled sensitivity test.")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(analyze(load_rows(args.dataset)))

if __name__ == "__main__":
    main()
