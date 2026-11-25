"""Treatment 3 race: learned configuration selector vs historical baseline."""
import csv
import math
import os
import sys
import time
from collections import defaultdict

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATASET = "moot/optimize/hpo/Health-ClosedIssues0000.csv"
HEADER = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
          "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
TARGETS = ["MRE-", "ACC+", "PRED40+"]
CONFIG = ["N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth"]


def load_validate(path=DATASET):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0]) != HEADER:
        raise ValueError("empty dataset or unexpected header")
    out = []
    for n, row in enumerate(rows, 2):
        item = dict(row)
        for k in HEADER:
            if k != "criterion":
                item[k] = float(row[k])
                if not math.isfinite(item[k]):
                    raise ValueError("non-finite value on row %d" % n)
        if any(not 0 <= item[k] <= 1 for k in TARGETS):
            raise ValueError("target outside [0,1] on row %d" % n)
        if item["N_estimators"] < 1 or item["Min_sample_leaves"] < 1 or item["Max_depth"] < 1:
            raise ValueError("invalid positive configuration on row %d" % n)
        out.append(item)
    return out


def utility(row):
    # Equal-weight quality utility: lower error and higher accuracy/prediction.
    return (1.0 - row["MRE-"] + row["ACC+"] + row["PRED40+"]) / 3.0


def make_model(seed):
    prep = ColumnTransformer([
        ("numeric", "passthrough", [0, 1, 2, 3]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), [4]),
    ])
    return Pipeline([("prep", prep), ("model", ExtraTreesRegressor(
        n_estimators=80, min_samples_leaf=4, random_state=seed, n_jobs=1))])


def choose_learned(train, candidates, seed):
    # Fit one multi-output regressor; all feature and target data are historical.
    model = make_model(seed)
    def features(rs):
        return [[r["N_estimators"], r["Min_sample_leaves"],
                 r["Min_impurity_decrease"], r["Max_depth"], r["criterion"]]
                for r in rs]
    model.fit(features(train),
              [[r[k] for k in TARGETS] for r in train])
    pred = model.predict(features(candidates))
    scores = [(1 - p[0] + p[1] + p[2]) / 3 for p in pred]
    return max(range(len(candidates)), key=lambda i: (scores[i], -i))


def choose_historical(train, candidates):
    # Historical strategy: mean utility of each exact configuration; unseen configs
    # receive the overall historical mean, making the fallback deterministic.
    groups = defaultdict(list)
    for r in train:
        groups[tuple(r[k] if k != "criterion" else r[k] for k in CONFIG)].append(utility(r))
    overall = sum(utility(r) for r in train) / len(train)
    estimates = []
    for i, r in enumerate(candidates):
        key = tuple(r[k] for k in CONFIG)
        estimates.append((sum(groups[key]) / len(groups[key]) if key in groups else overall, -i))
    return max(range(len(candidates)), key=lambda i: estimates[i])


def evaluate(rows, folds=5, seed=20261029):
    if len(rows) < folds + 1:
        raise ValueError("not enough rows")
    # Expanding chronological folds; each test block is never used for fitting.
    block = len(rows) // (folds + 1)
    records = []
    for fold in range(folds):
        train_end = block * (fold + 1)
        test_end = block * (fold + 2) if fold < folds - 1 else len(rows)
        train, test = rows[:train_end], rows[train_end:test_end]
        li = choose_learned(train, test, seed + fold)
        hi = choose_historical(train, test)
        for method, idx in (("treatment3_selector", li), ("historical_baseline", hi)):
            r = test[idx]
            records.append({"fold": fold + 1, "method": method,
                            "MRE-": r["MRE-"], "ACC+": r["ACC+"],
                            "PRED40+": r["PRED40+"], "utility": utility(r),
                            "train_rows": len(train), "test_rows": len(test)})
    summary = {}
    for method in ("treatment3_selector", "historical_baseline"):
        rs = [r for r in records if r["method"] == method]
        summary[method] = {k: sum(r[k] for r in rs) / len(rs)
                           for k in ("MRE-", "ACC+", "PRED40+", "utility")}
    return records, summary


def main(path=DATASET):
    started = time.perf_counter()
    rows = load_validate(path)
    records, summary = evaluate(rows)
    print("validated_rows=%d" % len(rows))
    print("protocol=5 expanding chronological folds; seed=20261029; utility=(1-MRE-+ACC++PRED40+)/3")
    for r in records:
        print("fold=%d method=%s MRE-=%.6f ACC+=%.6f PRED40+=%.6f utility=%.6f" %
              (r["fold"], r["method"], r["MRE-"], r["ACC+"], r["PRED40+"], r["utility"]))
    for m, s in summary.items():
        print("summary=%s %s" % (m, " ".join("%s=%.6f" % x for x in s.items())))
    winner = max(summary, key=lambda m: summary[m]["utility"])
    print("winner=%s" % winner)
    print("runtime_seconds=%.6f" % (time.perf_counter() - started))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET)
