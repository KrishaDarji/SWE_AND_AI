"""Step 10 (race): shared chooser vs random search, same budget, 10 repeats."""

from src.core.race import race


def run(spec, rows):
    return race(spec, rows, budget=20, repeats=10)
