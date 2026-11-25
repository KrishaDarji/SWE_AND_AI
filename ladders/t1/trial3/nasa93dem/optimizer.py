#!/usr/bin/env python3
"""Select NASA93 projects that maximize delivered value under an effort budget."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Project:
    project_id: int
    kloc: float
    effort: float
    defects: float

    @property
    def value(self) -> float:
        # Defects reduce the useful delivered size, while keeping value positive.
        return self.kloc / (1.0 + self.defects / 1000.0)


def load_projects(path: Path):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield Project(int(row["idX"]), float(row["Kloc+"]),
                          float(row["Effort-"]), float(row["Defects-"]))


def choose(projects, budget: float):
    """Return the highest-value feasible subset, using exact scaled DP."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    items = list(projects)
    scale = 100
    capacity = int(round(budget * scale))
    dp = [(0.0, ()) for _ in range(capacity + 1)]
    for index, project in enumerate(items):
        cost = int(round(project.effort * scale))
        if cost > capacity:
            continue
        for remaining in range(capacity, cost - 1, -1):
            previous_value, previous_indices = dp[remaining - cost]
            candidate = (previous_value + project.value, previous_indices + (index,))
            if candidate[0] > dp[remaining][0] + 1e-12:
                dp[remaining] = candidate
    value, indices = max(dp, key=lambda result: result[0])
    selected = [items[i] for i in indices]
    return selected, value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--budget", type=float, default=1000.0)
    args = parser.parse_args(argv)
    selected, value = choose(load_projects(args.dataset), args.budget)
    effort = sum(p.effort for p in selected)
    defects = sum(p.defects for p in selected)
    print(f"budget={args.budget:.2f} effort={effort:.2f} value={value:.4f}")
    print(f"selected_count={len(selected)} project_ids={','.join(str(p.project_id) for p in selected)}")
    print(f"kloc={sum(p.kloc for p in selected):.2f} defects={defects:.2f}")


if __name__ == "__main__":
    main()
