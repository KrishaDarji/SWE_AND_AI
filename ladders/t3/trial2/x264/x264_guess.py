"""Independent x264 encoding-time prediction skill.

The CSV is intentionally read with the standard-library csv module; pandas is
not required.  The final column is the measured encoding-time target.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = Path("moot/optimize/config/X264_AllMeasurements.csv")
TARGET = "Performance-"


def load_dataset(path: Path | str = DEFAULT_DATASET) -> Tuple[List[str], List[List[float]], List[float]]:
    """Load and validate numeric configuration rows and encoding-time target."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or TARGET not in reader.fieldnames:
            raise ValueError(f"missing target column {TARGET!r}")
        features = [name for name in reader.fieldnames if name != TARGET]
        if not features:
            raise ValueError("dataset has no configuration features")
        rows: List[List[float]] = []
        targets: List[float] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in features]
                target = float(row[TARGET])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric value at CSV line {line_number}") from exc
            if not all(math.isfinite(value) for value in values + [target]):
                raise ValueError(f"non-finite value at CSV line {line_number}")
            rows.append(values)
            targets.append(target)
    if len(rows) < 10:
        raise ValueError("dataset must contain at least 10 rows")
    return features, rows, targets


class X264GuessModel:
    """Reusable fitted predictor for x264 configuration encoding time."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.model = RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=1, min_samples_leaf=2
        )

    def fit(self, feature_names: Sequence[str], X: Sequence[Sequence[float]], y: Sequence[float]) -> "X264GuessModel":
        if len(X) != len(y) or not X:
            raise ValueError("X and y must be non-empty and have equal length")
        if any(len(row) != len(feature_names) for row in X):
            raise ValueError("feature row width does not match feature names")
        self.feature_names = list(feature_names)
        self.model.fit(X, y)
        return self

    def predict(self, configuration: Sequence[float] | Dict[str, float]) -> float:
        if not self.feature_names:
            raise RuntimeError("model is not fitted")
        if isinstance(configuration, dict):
            row = [float(configuration[name]) for name in self.feature_names]
        else:
            row = [float(value) for value in configuration]
        if len(row) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features")
        return float(self.model.predict([row])[0])


def evaluate(path: Path | str = DEFAULT_DATASET, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
    """Hold out data and report MAE, RMSE, and R2."""
    names, X, y = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    predictor = X264GuessModel(random_state).fit(names, X_train, y_train)
    prediction = predictor.model.predict(X_test)
    return {
        "rows": float(len(X)),
        "features": float(len(names)),
        "test_rows": float(len(X_test)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "r2": float(r2_score(y_test, prediction)),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict x264 encoding time")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predict", nargs="*", type=float, help="feature values in CSV order")
    args = parser.parse_args(argv)
    names, X, y = load_dataset(args.dataset)
    if args.predict is not None:
        predictor = X264GuessModel().fit(names, X, y)
        if len(args.predict) != len(names):
            parser.error(f"--predict requires {len(names)} values")
        print(f"prediction={predictor.predict(args.predict):.6f}")
    else:
        metrics = evaluate(args.dataset)
        print(f"dataset={args.dataset} rows={int(metrics['rows'])} features={int(metrics['features'])} test_rows={int(metrics['test_rows'])}")
        print(f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
