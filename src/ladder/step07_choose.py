"""Step 7 (choose): find a strong row while evaluating only 20 of them."""

from src.core.choose import choose


def run(spec, rows):
    return choose(spec, rows, budget=20)
