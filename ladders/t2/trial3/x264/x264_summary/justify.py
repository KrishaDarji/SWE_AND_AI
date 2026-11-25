"""Produce an evidence-based explanation for an x264 configuration."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class Justification:
    configuration: dict
    measure: str
    samples: int
    mean: float
    runner_up_mean: Optional[float]
    margin: Optional[float]
    recommendation: str


def justify_configuration(path: str, configuration: Optional[Mapping[str, object]] = None,
                          measure: str = "", higher_is_better: bool = False) -> Justification:
    """Recommend the best observed configuration, or justify an exact one.

    The recommendation is descriptive: it ranks observed configuration means and
    reports the margin over the runner-up. It does not claim causal evidence.
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
        if configuration is not None:
            missing = [name for name in settings if name not in configuration]
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
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    ordered = sorted(means, key=lambda key: (-means[key] if higher_is_better else means[key], key))
    selected = tuple(str(configuration[name]).strip() for name in settings) if configuration else ordered[0]
    if selected not in groups:
        raise ValueError("configuration must match at least one data row")
    ranked = sorted(means, key=lambda key: (-means[key] if higher_is_better else means[key], key))
    position = ranked.index(selected)
    runner = means[ranked[position + 1]] if position + 1 < len(ranked) else None
    mean = means[selected]
    margin = (mean - runner) if higher_is_better else (runner - mean) if runner is not None else None
    status = "recommended" if position == 0 else "not the top observed configuration"
    recommendation = (f"Configuration is {status}: mean {target}={mean:.6f} over "
                      f"{len(groups[selected])} sample(s)" +
                      (f", {margin:.6f} better than the runner-up." if margin is not None else "."))
    return Justification(dict(zip(settings, selected)), target, len(groups[selected]),
                         mean, runner, margin, recommendation)
