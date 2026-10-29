#!/usr/bin/env python3
"""Independent historical summary for the NASA93DEM project dataset."""
import argparse
import csv
import math
import os
import statistics
import tempfile
from typing import Dict, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

OUTCOMES = ["Kloc+", "Effort-", "Defects-", "MonthsX"]
ID_COLUMNS = ["idX"]


def load_dataset(path: str) -> Tuple[List[str], List[Dict[str, object]]]:
    if not os.path.isfile(path):
        raise ValueError("dataset does not exist: %s" % path)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(not name.strip() for name in reader.fieldnames):
            raise ValueError("CSV must have a non-empty header")
        fields = [name.strip() for name in reader.fieldnames]
        if len(set(fields)) != len(fields):
            raise ValueError("CSV header contains duplicate columns")
        rows = []
        for line_no, raw in enumerate(reader, 2):
            if None in raw or len(raw) != len(fields):
                raise ValueError("line %d has an invalid number of fields" % line_no)
            row = {}
            for field, value in zip(fields, raw.values()):
                value = value.strip()
                if not value:
                    raise ValueError("line %d field %s is blank" % (line_no, field))
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError
                    row[field] = number
                except ValueError:
                    row[field] = value
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return fields, rows


def identify_columns(fields: List[str]) -> Tuple[List[str], List[str], List[str]]:
    outcomes = [name for name in OUTCOMES if name in fields]
    if outcomes != OUTCOMES:
        raise ValueError("missing expected outcome columns: %s" % ",".join(set(OUTCOMES) - set(outcomes)))
    ids = [name for name in ID_COLUMNS if name in fields]
    features = [name for name in fields if name not in outcomes and name not in ids]
    if not features:
        raise ValueError("no project-feature columns identified")
    return features, outcomes, ids


def pearson(xs: List[float], ys: List[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return 0.0 if denominator == 0 else sum(x * y for x, y in zip(dx, dy)) / denominator


def analyze(path: str) -> str:
    fields, rows = load_dataset(path)
    features, outcomes, ids = identify_columns(fields)
    numeric_features = [f for f in features if all(isinstance(row[f], float) for row in rows)]
    categorical_features = [f for f in features if f not in numeric_features]
    lines = ["NASA93DEM remember analysis", "dataset=%s" % path,
             "rows=%d columns=%d" % (len(rows), len(fields)),
             "project_features=%s" % ",".join(features),
             "outcomes=%s" % ",".join(outcomes), "identifier_columns=%s" % ("none" if not ids else ",".join(ids)),
             "feature_types=numeric:%d categorical:%d" % (len(numeric_features), len(categorical_features))]
    for outcome in outcomes:
        values = [row[outcome] for row in rows]
        lines.append("outcome_summary %s min=%.3f max=%.3f mean=%.3f median=%.3f stdev=%.3f" %
                     (outcome, min(values), max(values), statistics.mean(values), statistics.median(values), statistics.stdev(values)))
    for feature in numeric_features:
        values = [row[feature] for row in rows]
        lines.append("numeric_feature %s min=%.3f max=%.3f mean=%.3f unique=%d" %
                     (feature, min(values), max(values), statistics.mean(values), len(set(values))))
    for feature in categorical_features:
        groups = {}
        for row in rows:
            groups.setdefault(str(row[feature]), []).append(row[feature])
        lines.append("categorical_feature %s levels=%s" % (feature, ",".join(sorted(groups))))
    for feature in numeric_features:
        for outcome in outcomes:
            lines.append("relationship pearson feature=%s outcome=%s r=%.4f" %
                         (feature, outcome, pearson([row[feature] for row in rows], [row[outcome] for row in rows])))
    for feature in categorical_features:
        levels = sorted({str(row[feature]) for row in rows})
        for outcome in outcomes:
            means = [statistics.mean([row[outcome] for row in rows if str(row[feature]) == level]) for level in levels]
            lines.append("relationship grouped_mean feature=%s outcome=%s %s" %
                         (feature, outcome, ";".join("%s=%.3f" % pair for pair in zip(levels, means))))
    # Use a temporary named CSV-like matrix without pandas: selectors are integer positions.
    cat_positions = [features.index(f) for f in categorical_features]
    num_positions = [features.index(f) for f in numeric_features]
    transformer = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_positions),
                                     ("num", "passthrough", num_positions)])
    x = transformer.fit_transform([[row[f] for f in features] for row in rows])
    model_scores = []
    for outcome in outcomes:
        y = [row[outcome] for row in rows]
        model = LinearRegression().fit(x, y)
        model_scores.append(r2_score(y, model.predict(x)))
    lines.append("one_hot_linear_regression train_r2 " + ",".join("%s=%.4f" % pair for pair in zip(outcomes, model_scores)))
    lines.append("assumptions=each row is a historical project; categorical levels are nominal; numeric fields are measured as supplied; outcome suffixes indicate direction only")
    lines.append("limitations=observational data and in-sample regression do not establish causation or validate future delivery; small sample and possible confounding limit subgroup comparisons")
    lines.append("data_quality=validated nonblank finite numeric cells; categorical cells retained verbatim after trimming; no missing values detected; source metadata semantics should be confirmed")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    print(analyze(parser.parse_args().dataset), end="")


if __name__ == "__main__":
    main()
