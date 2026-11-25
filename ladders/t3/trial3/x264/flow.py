"""Sequential, reproducible evaluation flow for the x264 benchmark."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class Evaluation:
    rows_seen: int
    valid_rows: int
    invalid_rows: int
    best_configuration: Optional[Tuple[int, ...]]
    best_performance: Optional[float]
    best_row: Optional[int]
    improvements: int


def _configuration_key(values):
    return tuple(values)


def evaluate(path):
    """Read one CSV record at a time and return the final running state."""
    rows_seen = valid_rows = invalid_rows = improvements = 0
    best_configuration = None
    best_performance = None
    best_row = None
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("dataset is empty") from exc
        if len(header) < 2 or header[-1] != "Performance-":
            raise ValueError("expected Performance- as the final column")
        names = header[:-1]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("configuration columns must be unique and non-empty")
        for row_number, row in enumerate(reader, start=2):
            rows_seen += 1
            if len(row) != len(header):
                invalid_rows += 1
                continue
            try:
                configuration = tuple(int(value) for value in row[:-1])
                performance = float(row[-1])
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            if any(value not in (0, 1) for value in configuration) or not math.isfinite(performance):
                invalid_rows += 1
                continue
            valid_rows += 1
            candidate = _configuration_key(configuration)
            if (best_performance is None or performance < best_performance or
                    (performance == best_performance and candidate < best_configuration)):
                best_configuration = candidate
                best_performance = performance
                best_row = row_number
                improvements += 1
    return Evaluation(rows_seen, valid_rows, invalid_rows, best_configuration,
                      best_performance, best_row, improvements)


def format_report(result):
    return (f"rows_seen={result.rows_seen}\nvalid_rows={result.valid_rows}\n"
            f"invalid_rows={result.invalid_rows}\nimprovements={result.improvements}\n"
            f"best_row={result.best_row}\nbest_performance={result.best_performance}\n"
            f"best_configuration={result.best_configuration}")


if __name__ == "__main__":
    import sys
    print(format_report(evaluate(sys.argv[1])))
