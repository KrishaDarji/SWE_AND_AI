#!/usr/bin/env python3
"""Summarize the x264 historical measurements without third-party data tools."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DataError(ValueError):
    pass


def _finite_float(value: str, row: int, column: str) -> float:
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise DataError(f"row {row}: {column!r} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DataError(f"row {row}: {column!r} is not finite")
    return number


def load_dataset(path: Path) -> Tuple[List[str], List[List[float]]]:
    if not path.is_file():
        raise DataError(f"dataset does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration as exc:
            raise DataError("dataset is empty") from exc
        if not header or any(not cell for cell in header):
            raise DataError("header contains an empty column name")
        if len(set(header)) != len(header):
            raise DataError("header contains duplicate column names")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise DataError(f"row {row_number}: expected {len(header)} fields, got {len(row)}")
            rows.append([_finite_float(value, row_number, name) for name, value in zip(header, row)])
    if not rows:
        raise DataError("dataset has no data rows")
    return header, rows


def identify_columns(header: Sequence[str], rows: Sequence[Sequence[float]]) -> Tuple[List[str], str]:
    performance = [name for name in header if "performance" in name.lower() or "throughput" in name.lower()]
    if len(performance) != 1:
        raise DataError(f"expected exactly one performance column, found {performance}")
    target = performance[0]
    settings = [name for name in header if name != target]
    if not settings:
        raise DataError("no configuration columns found")
    index = {name: i for i, name in enumerate(header)}
    for name in settings:
        values = {row[index[name]] for row in rows}
        if not values:
            raise DataError(f"configuration column {name!r} is empty")
    return settings, target


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    sx, sy = statistics.mean(x), statistics.mean(y)
    dx = [a - sx for a in x]
    dy = [b - sy for b in y]
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else 0.0


def summarize(path: Path) -> str:
    header, rows = load_dataset(path)
    settings, target = identify_columns(header, rows)
    pos = {name: i for i, name in enumerate(header)}
    y = [row[pos[target]] for row in rows]
    x = [[row[pos[name]] for name in settings] for row in rows]
    # Ridge keeps coefficients finite when the one-hot-like settings are collinear.
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="lsqr")).fit(x, y)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    r2 = cross_val_score(model, x, y, cv=cv, scoring="r2")
    correlations = [(name, pearson([row[pos[name]] for row in rows], y)) for name in settings]
    correlations.sort(key=lambda pair: abs(pair[1]), reverse=True)
    lines = [
        "x264 historical dataset summary",
        f"dataset: {path}",
        f"rows: {len(rows)}; columns: {len(header)}",
        f"configuration columns ({len(settings)}): {', '.join(settings)}",
        f"performance column: {target}",
        f"performance: min={min(y):.3f}, mean={statistics.mean(y):.3f}, median={statistics.median(y):.3f}, max={max(y):.3f}, stdev={statistics.stdev(y):.3f}",
        "setting summaries (unique values; mean performance when enabled/value is present):",
    ]
    for name in settings:
        values = sorted(set(row[pos[name]] for row in rows))
        groups = []
        for value in values:
            selected = [y[i] for i, row in enumerate(rows) if row[pos[name]] == value]
            groups.append(f"{value:g}->{statistics.mean(selected):.3f} (n={len(selected)})")
        lines.append(f"  {name}: values={','.join(f'{v:g}' for v in values)}; " + "; ".join(groups))
    lines.append("strongest marginal Pearson relationships (absolute r descending):")
    for name, corr in correlations:
        lines.append(f"  {name}: r={corr:+.4f}")
    coefficients = dict(zip(settings, model[-1].coef_))
    lines.append("standardized multivariate linear coefficients (directional association, not causation):")
    for name, coefficient in sorted(coefficients.items(), key=lambda p: abs(p[1]), reverse=True):
        lines.append(f"  {name}: beta={coefficient:+.4f}")
    lines.append(f"training R2={model.score(x, y):.4f}; 5-fold shuffled CV R2 mean={statistics.mean(r2):.4f}, stdev={statistics.stdev(r2):.4f}")
    lines += [
        "assumptions and limitations:",
        "  Numeric columns are treated as measured settings; equal numeric values are grouped exactly.",
        "  Performance is assumed to be a larger-is-better metric because its name is Performance-; the CSV does not state units.",
        "  Correlation and linear coefficients describe association in this historical sample and do not establish causal effects or interactions.",
        "  Possible data-quality risks include undocumented units, duplicate configurations, measurement noise, and the absence of an explicit experiment protocol.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(summarize(args.dataset), end="")


if __name__ == "__main__":
    main()
