"""Choose the best observed configuration subject to a quality constraint."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Choice:
    configuration: dict
    quality_measure: str
    quality_mean: float
    objective_measure: str
    objective_mean: float
    samples: int
    feasible_configurations: int


def choose_configuration(path: str, quality_measure: str, minimum_quality: Optional[float] = None,
                         maximum_quality: Optional[float] = None, objective_measure: str = "",
                         quality_higher_is_better: bool = True,
                         objective_higher_is_better: bool = False) -> Choice:
    """Select the best observed configuration whose quality mean is in bounds."""
    if minimum_quality is None and maximum_quality is None:
        raise ValueError("at least one quality bound is required")
    if minimum_quality is not None and maximum_quality is not None and minimum_quality > maximum_quality:
        raise ValueError("minimum quality cannot exceed maximum quality")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        headers = list(reader.fieldnames)
        if quality_measure not in headers:
            raise ValueError(f"unknown quality measure column: {quality_measure}")
        objective = objective_measure or quality_measure
        if objective not in headers:
            raise ValueError(f"unknown objective measure column: {objective}")
        settings = [name for name in headers if name not in {quality_measure, objective}]
        groups = {}
        for row_number, row in enumerate(reader, 2):
            try:
                quality = float(str(row[quality_measure]).strip())
                value = float(str(row[objective]).strip())
            except ValueError as exc:
                raise ValueError(f"row {row_number}: measure is not numeric") from exc
            key = tuple(row[name].strip() for name in settings)
            groups.setdefault(key, [[], []])
            groups[key][0].append(quality)
            groups[key][1].append(value)
    if not groups:
        raise ValueError("CSV must contain at least one data row")
    feasible = []
    for key, (qualities, objectives) in groups.items():
        quality_mean = sum(qualities) / len(qualities)
        if minimum_quality is not None and quality_mean < minimum_quality:
            continue
        if maximum_quality is not None and quality_mean > maximum_quality:
            continue
        feasible.append((key, quality_mean, sum(objectives) / len(objectives), len(qualities)))
    if not feasible:
        raise ValueError("no configuration satisfies the quality constraint")
    if objective_higher_is_better:
        chosen = min(feasible, key=lambda item: (-item[2], item[0]))
    else:
        chosen = min(feasible, key=lambda item: (item[2], item[0]))
    key, quality_mean, objective_mean, samples = chosen
    return Choice(dict(zip(settings, key)), quality_measure, quality_mean, objective,
                  objective_mean, samples, len(feasible))
