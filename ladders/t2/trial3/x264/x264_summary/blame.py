"""Rank settings associated with slower encode measurements."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Blame:
    setting: str
    enabled_n: int
    disabled_n: int
    enabled_mean: float
    disabled_mean: float
    penalty: float


def blame_slow_encodes(path: str, measure: str = "") -> List[Blame]:
    """Rank binary settings by enabled-minus-disabled mean encode time.

    Positive ``penalty`` means the enabled setting is associated with slower
    encodes. Rows with values other than 0/1 are rejected for settings.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        headers = list(reader.fieldnames)
        target = measure or headers[-1]
        if target not in headers:
            raise ValueError(f"unknown measure column: {target}")
        settings = [name for name in headers if name != target]
        groups = {name: {"0": [], "1": []} for name in settings}
        for row_number, row in enumerate(reader, 2):
            try:
                value = float(str(row[target]).strip())
            except ValueError as exc:
                raise ValueError(f"row {row_number}: {target!r} is not numeric") from exc
            for name in settings:
                state = row[name].strip()
                if state not in groups[name]:
                    raise ValueError(f"row {row_number}: {name!r} must be 0 or 1")
                groups[name][state].append(value)
    result = []
    for name in settings:
        enabled, disabled = groups[name]["1"], groups[name]["0"]
        if not enabled or not disabled:
            continue
        enabled_mean = sum(enabled) / len(enabled)
        disabled_mean = sum(disabled) / len(disabled)
        result.append(Blame(name, len(enabled), len(disabled), enabled_mean,
                            disabled_mean, enabled_mean - disabled_mean))
    result.sort(key=lambda item: (-item.penalty, item.setting))
    return result
