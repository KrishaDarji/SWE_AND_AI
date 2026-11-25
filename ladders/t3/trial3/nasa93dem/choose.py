"""Select a NASA93DEM project configuration under a fixed effort budget."""

import csv
from pathlib import Path


REQUIRED = ("idX", "Kloc+", "Effort-", "Defects-", "MonthsX")
BUDGET = 500.0


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(column not in rows[0] for column in REQUIRED):
        raise ValueError("dataset is empty or missing required columns")
    for row in rows:
        for column in REQUIRED:
            if column == "idX":
                int(row[column])
            else:
                value = float(row[column])
                if value < 0:
                    raise ValueError(f"negative value in {column}")
    return rows


def delivered_value(row):
    """Value proxy: delivered KLOC discounted by defects per KLOC."""
    kloc = float(row["Kloc+"])
    defects = float(row["Defects-"])
    return kloc / (1.0 + defects / max(kloc, 1e-12))


def choose(rows, budget=BUDGET):
    feasible = [row for row in rows if float(row["Effort-"]) <= budget]
    if not feasible:
        return None, []
    ranked = sorted(feasible, key=lambda row: (-delivered_value(row), float(row["Effort-"]), int(row["idX"])))
    return ranked[0], ranked


def main(dataset_path):
    rows = load_dataset(dataset_path)
    selected, feasible = choose(rows)
    if selected is None:
        return {"dataset_rows": len(rows), "feasible_count": 0, "selected": None}
    return {
        "dataset_rows": len(rows),
        "feasible_count": len(feasible),
        "selected": {
            "id": int(selected["idX"]),
            "effort": float(selected["Effort-"]),
            "value": delivered_value(selected),
            "kloc": float(selected["Kloc+"]),
            "defects": float(selected["Defects-"]),
            "months": float(selected["MonthsX"]),
        },
    }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(main(args.dataset), indent=2, sort_keys=True))
