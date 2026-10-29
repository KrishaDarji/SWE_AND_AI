"""Streaming proposal evaluation for NASA93DEM.

The flow minimizes Effort- and breaks equal-effort ties by numeric idX,
then arrival order.  It never loads the data rows as a batch.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


REQUIRED_COLUMNS = ("idX", "Kloc+", "Effort-", "Defects-", "MonthsX")


@dataclass(frozen=True)
class Evaluation:
    rows_seen: int
    valid_rows: int
    invalid_rows: int
    incomplete_rows: int
    improvements: int
    best_proposal: Optional[Tuple[str, ...]]
    best_row: Optional[int]
    best_effort: Optional[float]


def _valid_number(value):
    number = float(value.strip())
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def _proposal_key(row, arrival):
    # idX is the stable proposal identifier; arrival is the final deterministic
    # tie-breaker when a file contains duplicate ids and equal effort.
    try:
        identifier = int(row[0].strip())
    except ValueError:
        identifier = math.inf
    return (identifier, arrival)


def evaluate(path) -> Evaluation:
    rows_seen = valid_rows = invalid_rows = incomplete_rows = improvements = 0
    best_proposal = None
    best_row = None
    best_effort = None
    best_key = None

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("dataset is empty") from exc
        if len(header) != len(set(header)) or any(not name.strip() for name in header):
            raise ValueError("header names must be unique and non-empty")
        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise ValueError("missing required columns: " + ",".join(missing))
        effort_index = header.index("Effort-")
        id_index = header.index("idX")

        for row_number, row in enumerate(reader, start=2):
            rows_seen += 1
            if len(row) < len(header):
                invalid_rows += 1
                incomplete_rows += 1
                continue
            if len(row) != len(header):
                invalid_rows += 1
                continue
            try:
                if any(not value.strip() for value in row):
                    raise ValueError("blank field")
                effort = _valid_number(row[effort_index])
                _valid_number(row[id_index])
                for column in ("Kloc+", "Defects-", "MonthsX"):
                    _valid_number(row[header.index(column)])
            except (TypeError, ValueError):
                invalid_rows += 1
                continue

            valid_rows += 1
            key = _proposal_key(row, valid_rows)
            if (best_effort is None or effort < best_effort or
                    (effort == best_effort and key < best_key)):
                best_effort = effort
                best_proposal = tuple(row)
                best_row = row_number
                best_key = key
                improvements += 1

    return Evaluation(rows_seen, valid_rows, invalid_rows, incomplete_rows,
                      improvements, best_proposal, best_row, best_effort)


def format_report(result):
    proposal_id = result.best_proposal[0] if result.best_proposal else None
    return (f"rows_seen={result.rows_seen}\nvalid_rows={result.valid_rows}\n"
            f"invalid_rows={result.invalid_rows}\nincomplete_rows={result.incomplete_rows}\n"
            f"improvements={result.improvements}\nbest_row={result.best_row}\n"
            f"best_idX={proposal_id}\nbest_effort={result.best_effort}")


if __name__ == "__main__":
    import sys
    print(format_report(evaluate(sys.argv[1])))
