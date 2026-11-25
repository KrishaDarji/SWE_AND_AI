"""Step 9 (spot): flag rows that look unlike the rest of the data."""

from src.core.spot import spot


def run(spec, rows):
    return {"findings": spot(spec, rows, top_k=5)}
