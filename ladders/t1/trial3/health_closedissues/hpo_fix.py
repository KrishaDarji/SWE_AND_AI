"""Find the least-changing observed configuration with a lower model error."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ERROR_COLUMN = "MRE-"
PARAMETERS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
NUMERIC_PARAMETERS = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}


@dataclass(frozen=True)
class Config:
    error: float
    values: dict[str, Any]
    row_number: int


def parse_value(name: str, value: str) -> Any:
    if name in {"N_estimators", "Min_sample_leaves", "Max_depth"}:
        return int(value)
    if name == "Min_impurity_decrease":
        return float(value)
    return value


def load_configs(path: Path) -> list[Config]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {ERROR_COLUMN, *PARAMETERS}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"missing required columns: {sorted(required - set(reader.fieldnames or ())) }")
        configs = []
        for row_number, row in enumerate(reader, start=2):
            values = {name: parse_value(name, row[name]) for name in PARAMETERS}
            configs.append(Config(float(row[ERROR_COLUMN]), values, row_number))
    if not configs:
        raise ValueError("dataset contains no configurations")
    return configs


def change_count(base: Config, candidate: Config) -> int:
    return sum(base.values[name] != candidate.values[name] for name in PARAMETERS)


def suggest(configs: Iterable[Config]) -> tuple[Config, Config, int]:
    rows = list(configs)
    baseline = rows[0]
    better = [row for row in rows[1:] if row.error < baseline.error]
    if not better:
        raise ValueError("no observed configuration improves the baseline error")
    chosen = min(better, key=lambda row: (change_count(baseline, row), row.error, row.row_number))
    return baseline, chosen, change_count(baseline, chosen)


def format_result(baseline: Config, chosen: Config, changes: int) -> str:
    changed = [name for name in PARAMETERS if baseline.values[name] != chosen.values[name]]
    details = ", ".join(f"{name}: {baseline.values[name]} -> {chosen.values[name]}" for name in changed)
    improvement = baseline.error - chosen.error
    return (f"Baseline row {baseline.row_number}: MRE={baseline.error:.12g}\n"
            f"Suggested row {chosen.row_number}: MRE={chosen.error:.12g}\n"
            f"Change {changes} hyperparameter(s): {details}\n"
            f"Absolute MRE improvement: {improvement:.12g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    baseline, chosen, changes = suggest(load_configs(args.dataset))
    print(format_result(baseline, chosen, changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
