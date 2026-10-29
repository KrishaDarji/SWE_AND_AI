"""Independent configuration-fix evaluation for Health-ClosedIssues."""
import csv
from pathlib import Path

CONFIG = ("N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth")
METRICS = ("MRE-", "ACC+", "PRED40+")


def load_validate(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(CONFIG + METRICS)
    if not rows or not required.issubset(rows[0]):
        raise ValueError("missing required columns or empty dataset")
    for row in rows:
        for column in METRICS:
            value = float(row[column])
            if value < 0:
                raise ValueError("metrics must be non-negative")
        for column in CONFIG:
            if not row[column]:
                raise ValueError("configuration values must be non-empty")
    return rows


def distance(a, b):
    return sum(a[key] != b[key] for key in CONFIG)


def choose(rows):
    baseline = rows[0]
    candidates = [
        row for row in rows[1:]
        if float(row["MRE-"]) < float(baseline["MRE-"])
        and float(row["ACC+"]) >= float(baseline["ACC+"])
        and float(row["PRED40+"]) >= float(baseline["PRED40+"])
    ]
    if not candidates:
        raise ValueError("no quality-preserving improvement found")
    minimum = min(distance(row, baseline) for row in candidates)
    candidates = [row for row in candidates if distance(row, baseline) == minimum]
    revised = min(candidates, key=lambda row: (float(row["MRE-"]),
                                                -float(row["ACC+"]),
                                                -float(row["PRED40+"])))
    return baseline, revised, minimum


def main(dataset):
    rows = load_validate(dataset)
    baseline, revised, changes = choose(rows)
    print("dataset_rows=%d" % len(rows))
    print("baseline_index=0")
    print("one_change_improvements=%d" % sum(
        distance(row, baseline) == 1 and float(row["MRE-"]) < float(baseline["MRE-"])
        and float(row["ACC+"]) >= float(baseline["ACC+"])
        and float(row["PRED40+"]) >= float(baseline["PRED40+"]) for row in rows[1:]))
    print("selected_changes=%d" % changes)
    print("baseline=" + ",".join("%s=%s" % (k, baseline[k]) for k in METRICS + CONFIG))
    print("revised=" + ",".join("%s=%s" % (k, revised[k]) for k in METRICS + CONFIG))
    print("mre_improvement=%.12f" % (float(baseline["MRE-"]) - float(revised["MRE-"])))
    print("acc_change=%.12f" % (float(revised["ACC+"]) - float(baseline["ACC+"])))
    print("pred40_change=%.12f" % (float(revised["PRED40+"]) - float(baseline["PRED40+"])))


if __name__ == "__main__":
    main("moot/optimize/hpo/Health-ClosedIssues0000.csv")
