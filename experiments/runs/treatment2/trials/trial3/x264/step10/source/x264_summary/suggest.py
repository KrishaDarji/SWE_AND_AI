"""Suggest the best one-setting change from a slow observed configuration."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class Suggestion:
    baseline: dict
    candidate: dict
    setting: str
    baseline_mean: float
    candidate_mean: float
    improvement: float
    baseline_samples: int
    candidate_samples: int


def suggest_setting_change(path: str, baseline: Mapping[str, object],
                           measure: str = "", higher_is_better: bool = False) -> Suggestion:
    """Return the fastest observed configuration differing in exactly one setting.

    For the default encoding-time objective, ``improvement`` is baseline mean minus
    candidate mean. The candidate must be fully observed in the CSV.
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
        missing = [name for name in settings if name not in baseline]
        if missing:
            raise ValueError("missing setting(s): " + ", ".join(missing))
        groups = {}
        for row_number, row in enumerate(reader, 2):
            try:
                value = float(str(row[target]).strip())
            except ValueError as exc:
                raise ValueError(f"row {row_number}: {target!r} is not numeric") from exc
            key = tuple(row[name].strip() for name in settings)
            groups.setdefault(key, []).append(value)
    if not groups:
        raise ValueError("CSV must contain at least one data row")
    base_key = tuple(str(baseline[name]).strip() for name in settings)
    if base_key not in groups:
        raise ValueError("baseline must match at least one data row")
    candidates = [(key, values) for key, values in groups.items()
                  if sum(a != b for a, b in zip(key, base_key)) == 1]
    if not candidates:
        raise ValueError("no observed one-setting change from baseline")
    mean = lambda values: sum(values) / len(values)
    base_mean = mean(groups[base_key])
    if higher_is_better:
        key, values = min(candidates, key=lambda item: (-mean(item[1]), item[0]))
        improvement = mean(values) - base_mean
    else:
        key, values = min(candidates, key=lambda item: (mean(item[1]), item[0]))
        improvement = base_mean - mean(values)
    changed = next(i for i, (a, b) in enumerate(zip(base_key, key)) if a != b)
    return Suggestion(dict(zip(settings, base_key)), dict(zip(settings, key)),
                      settings[changed], base_mean, mean(values), improvement,
                      len(groups[base_key]), len(values))
