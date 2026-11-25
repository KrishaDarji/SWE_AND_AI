"""CSV persistence boundary; no third-party dependencies are required."""

import csv
from pathlib import Path
from typing import Iterator, Union

from .model import Project


def _text(row: dict[str, str], name: str) -> str:
    return row.get(name, "").strip()


def _integer(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decimal(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CsvProjectRepository:
    """Lazily reads projects from a CSV path, isolating file format concerns."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def projects(self) -> Iterator[Project]:
        with self.path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield Project(
                    project_id=_text(row, "idX"),
                    year=_integer(_text(row, "YearX")),
                    kloc=_decimal(_text(row, "Kloc+")),
                    effort=_decimal(_text(row, "Effort-")),
                    defects=_decimal(_text(row, "Defects-")),
                    months=_decimal(_text(row, "MonthsX")),
                    attributes={key: value.strip() for key, value in row.items()
                                if key not in {"idX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}},
                )
