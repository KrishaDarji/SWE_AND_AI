"""Evidence-based comparison of the cost of two historical processes."""

import random
from statistics import median
from typing import Iterable, Optional

from .model import Project


def _costs(projects: Iterable[Project], attribute: str, value: str) -> list[float]:
    return [item.effort / item.kloc for item in projects
            if item.attributes.get(attribute, "").strip() == value
            and item.effort is not None and item.kloc is not None and item.kloc > 0]


def certify_less_costly(projects: Iterable[Project], attribute: str,
                        process_a: str, process_b: str, *, trials: int = 10000,
                        seed: int = 0) -> dict[str, object]:
    """Compare effort/KLOC and certify A as cheaper only at p < .05.

    A one-sided permutation test is used because the dataset is small and no
    distributional assumption is required.  The returned p-value is an
    estimate when ``trials`` is finite and is reproducible via ``seed``.
    """
    if trials <= 0:
        raise ValueError("trials must be greater than zero")
    if process_a == process_b:
        raise ValueError("processes must be different")
    first, second = _costs(projects, attribute, process_a), _costs(projects, attribute, process_b)
    if not first or not second:
        return {"attribute": attribute, "process_a": process_a, "process_b": process_b,
                "sample_a": len(first), "sample_b": len(second), "median_a": None,
                "median_b": None, "difference": None, "p_value": None,
                "verdict": "insufficient data"}
    observed = median(second) - median(first)
    combined = first + second
    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        difference = median(shuffled[len(first):]) - median(shuffled[:len(first)])
        extreme += difference >= observed
    p_value = (extreme + 1) / (trials + 1)
    verdict = "process A is genuinely less costly" if observed > 0 and p_value < 0.05 else "no genuine cost advantage established"
    return {"attribute": attribute, "process_a": process_a, "process_b": process_b,
            "sample_a": len(first), "sample_b": len(second), "median_a": median(first),
            "median_b": median(second), "difference": observed, "p_value": p_value,
            "verdict": verdict}
