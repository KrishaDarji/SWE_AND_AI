"""Independent Treatment 3 estimator and evaluation runner."""
import csv
import math
import os
import sys
import time
from statistics import median

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Effort-"
DATASET = os.path.join("moot", "optimize", "process", "nasa93dem.csv")
SEED = 20261029


def load_dataset(path=DATASET):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {TARGET, "Kloc+", "idX"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("missing required columns or empty dataset")
    for row in rows:
        if not row[TARGET].strip() or float(row[TARGET]) <= 0:
            raise ValueError("invalid effort")
        float(row["Kloc+"])
    return rows


def make_matrix(rows):
    xcols = [c for c in rows[0] if c not in {TARGET, "Defects-", "MonthsX", "idX"}]
    cats = [i for i, c in enumerate(xcols) if not _is_number(rows[0][c])]
    nums = [i for i, c in enumerate(xcols) if i not in cats]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", StandardScaler(), nums),
    ])
    return prep.fit_transform([[r[c] for c in xcols] for r in rows]), xcols, prep


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class RaceEstimator:
    """Robust Analogical Cost Estimator (RACE), fit only on historical rows."""
    def __init__(self, neighbors=7):
        self.neighbors = neighbors

    def fit(self, rows):
        self.rows = rows
        self.x, self.xcols, self.prep = make_matrix(rows)
        self.y = [float(r[TARGET]) for r in rows]
        self.nn = NearestNeighbors(n_neighbors=min(self.neighbors, len(rows)), metric="euclidean").fit(self.x)
        return self

    def predict(self, rows):
        q = self.prep.transform([[r[c] for c in self.xcols] for r in rows])
        distances, indexes = self.nn.kneighbors(q)
        out = []
        for ds, ix in zip(distances, indexes):
            weights = [1.0 / max(float(d), 1e-9) for d in ds]
            out.append(sum(w * self.y[i] for w, i in zip(weights, ix)) / sum(weights))
        return out


def historical_baseline(train, test):
    value = median(float(r[TARGET]) for r in train)
    return [value] * len(test)


def metrics(actual, predicted):
    return {"mae": mean_absolute_error(actual, predicted),
            "rmse": math.sqrt(mean_squared_error(actual, predicted)),
            "r2": r2_score(actual, predicted)}


def evaluate(rows):
    actual_all = []
    race_all = []
    base_all = []
    splitter = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    for train_i, test_i in splitter.split(rows):
        train = [rows[i] for i in train_i]
        test = [rows[i] for i in test_i]
        actual_all.extend(float(r[TARGET]) for r in test)
        race_all.extend(RaceEstimator().fit(train).predict(test))
        base_all.extend(historical_baseline(train, test))
    return metrics(actual_all, race_all), metrics(actual_all, base_all), len(actual_all)


def main():
    start = time.perf_counter()
    path = sys.argv[1] if len(sys.argv) > 1 else DATASET
    rows = load_dataset(path)
    race, base, n = evaluate(rows)
    print("dataset_rows=%d evaluation_predictions=%d seed=%d" % (len(rows), n, SEED))
    for name, result in (("race", race), ("historical_baseline", base)):
        print("%s mae=%.6f rmse=%.6f r2=%.6f" % (name, result["mae"], result["rmse"], result["r2"]))
    better = "race" if race["mae"] < base["mae"] else "historical_baseline"
    print("better_by_mae=%s" % better)
    print("runtime_seconds=%.6f" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
