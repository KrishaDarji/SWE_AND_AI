"""Incremental evaluation of benchmark results arriving one row at a time."""

import math
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class Evaluation:
    configuration: dict
    samples: int
    mean: float
    best: bool


class StreamingEvaluator:
    """Maintain configuration means without requiring the full dataset in memory."""

    def __init__(self, setting_columns, measure, higher_is_better=False):
        self.setting_columns = tuple(setting_columns)
        self.measure = measure
        self.higher_is_better = higher_is_better
        self._groups = {}
        self._order = []

    def add(self, row: Mapping[str, object]) -> Evaluation:
        missing = [name for name in self.setting_columns + (self.measure,)
                   if name not in row]
        if missing:
            raise ValueError("missing column(s): " + ", ".join(missing))
        try:
            value = float(str(row[self.measure]).strip())
        except ValueError as exc:
            raise ValueError("measure must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError("measure must be finite")
        configuration = tuple(str(row[name]).strip() for name in self.setting_columns)
        if configuration not in self._groups:
            self._groups[configuration] = [0, 0.0]
            self._order.append(configuration)
        group = self._groups[configuration]
        group[0] += 1
        group[1] += value
        best = self.best_configuration()
        return Evaluation(dict(zip(self.setting_columns, configuration)), group[0],
                          group[1] / group[0], best == configuration)

    def best_configuration(self) -> Optional[tuple]:
        if not self._groups:
            return None
        if self.higher_is_better:
            key = lambda item: (-item[1][1] / item[1][0], self._order.index(item[0]))
        else:
            key = lambda item: (item[1][1] / item[1][0], self._order.index(item[0]))
        return min(self._groups.items(), key=key)[0]

    def results(self):
        best = self.best_configuration()
        return [Evaluation(dict(zip(self.setting_columns, config)), self._groups[config][0],
                           self._groups[config][1] / self._groups[config][0], config == best)
                for config in self._order]
