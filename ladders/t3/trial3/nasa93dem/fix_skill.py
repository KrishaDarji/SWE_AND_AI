"""Recommend the cheapest one-factor process change that lowers NASA93DEM risk."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


PROCESS_COLUMNS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
                   "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
                   "apex", "plex", "ltex", "tool", "site", "sced")
REQUIRED = ("idX", "Kloc+", "Defects-") + PROCESS_COLUMNS
LEVELS = ("vl", "l", "n", "h", "vh", "xh")


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(c not in rows[0] for c in REQUIRED):
        raise ValueError("dataset is empty or missing required columns")
    for row in rows:
        int(row["idX"])
        if float(row["Kloc+"]) <= 0 or float(row["Defects-"]) < 0:
            raise ValueError("invalid Kloc+ or Defects- value")
        for c in PROCESS_COLUMNS:
            if not row[c].strip():
                raise ValueError(f"missing process value in {c}")
    return rows


def risk(row):
    """Observed risk proxy: defects per delivered KLOC."""
    return float(row["Defects-"]) / float(row["Kloc+"])


def modal_configuration(rows):
    return {c: Counter(row[c].strip() for row in rows).most_common(1)[0][0]
            for c in PROCESS_COLUMNS}


def _model(rows):
    x = [{c: row[c].strip() for c in PROCESS_COLUMNS} for row in rows]
    y = [risk(row) for row in rows]
    pipe = Pipeline([("encode", DictVectorizer(sparse=False)),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=3,
                                             min_samples_leaf=3))])
    pipe.fit(x, y)
    return pipe


def recommend(rows):
    model = _model(rows)
    baseline = modal_configuration(rows)
    baseline_risk = float(model.predict([baseline])[0])
    candidates = []
    for column in PROCESS_COLUMNS:
        observed = sorted({row[column].strip() for row in rows}, key=lambda v: (LEVELS.index(v), v))
        current = baseline[column]
        for value in observed:
            if value == current:
                continue
            candidate = dict(baseline)
            candidate[column] = value
            estimated = float(model.predict([candidate])[0])
            distance = abs(LEVELS.index(value) - LEVELS.index(current))
            candidates.append({"factor": column, "from": current, "to": value,
                               "cost": float(distance), "risk": estimated,
                               "reduction": baseline_risk - estimated})
    beneficial = [c for c in candidates if c["reduction"] > 1e-9]
    beneficial.sort(key=lambda c: (c["cost"], -c["reduction"], c["factor"], c["to"]))
    return {"baseline": baseline, "baseline_risk": baseline_risk,
            "candidate_count": len(candidates),
            "recommendation": beneficial[0] if beneficial else None}


def main(dataset_path):
    rows = load_dataset(dataset_path)
    result = recommend(rows)
    result["dataset_rows"] = len(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(main(args.dataset), indent=2, sort_keys=True))
