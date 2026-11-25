"""Independent NASA93DEM software-project effort estimator."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Effort-"
EXCLUDED = {"idX", "Defects-", "MonthsX"}

def _value(v: Any) -> Any:
    try: return float(v)
    except (TypeError, ValueError): return str(v).strip()

def load_dataset(path: str | Path):
    """Load, normalize, and validate the CSV."""
    p = Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or TARGET not in reader.fieldnames: raise ValueError("missing Effort- target")
        rows = [{k: str(v).strip() for k, v in row.items()} for row in reader]
    if len(rows) < 10: raise ValueError("dataset is too small")
    for i, row in enumerate(rows, 2):
        try: row[TARGET] = float(row[TARGET])
        except (KeyError, ValueError) as e: raise ValueError(f"invalid target at row {i}") from e
    return rows, list(reader.fieldnames)

def _features(fields: Sequence[str]):
    result = [x for x in fields if x != TARGET and x not in EXCLUDED]
    if not result: raise ValueError("no project features")
    return result

def _pipeline(rows, fields, random_state=42):
    names = _features(fields)
    X = [[_value(row[n]) for n in names] for row in rows]
    y = [float(row[TARGET]) for row in rows]
    numeric = [i for i in range(len(names)) if all(isinstance(x[i], float) for x in X)]
    categorical = [i for i in range(len(names)) if i not in numeric]
    transformers = []
    if numeric: transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    model = Pipeline([("preprocess", ColumnTransformer(transformers)), ("regressor", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=1))])
    return model, names, X, y

def train_model(rows, fields, random_state=42):
    """Fit and return (model, feature_names) for reusable predictions."""
    model, names, X, y = _pipeline(rows, fields, random_state)
    fitted = model.fit(X, y)
    fitted._guess_feature_names = names
    return fitted, names

def evaluate_model(rows, fields, test_size=.2, random_state=42):
    """Evaluate a fixed holdout and return measured regression metrics."""
    model, names, X, y = _pipeline(rows, fields, random_state)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(Xtr, ytr); pred = model.predict(Xte)
    return {"n_rows": len(rows), "n_features": len(names), "test_rows": len(yte), "mae": mean_absolute_error(yte, pred), "rmse": math.sqrt(mean_squared_error(yte, pred)), "r2": r2_score(yte, pred)}

def predict_effort(model, project: Mapping[str, Any]) -> float:
    """Predict effort for a proposal using a model returned by train_model."""
    names = getattr(model, "_guess_feature_names", list(project))
    return float(model.predict([[_value(project[n]) for n in names]])[0])

def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Estimate NASA93DEM project effort")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    rows, fields = load_dataset(args.dataset)
    print(json.dumps(evaluate_model(rows, fields), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
