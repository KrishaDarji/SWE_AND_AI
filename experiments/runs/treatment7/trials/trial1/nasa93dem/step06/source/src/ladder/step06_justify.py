"""Step 6 (justify): explain the top-ranked row using the blame evidence."""

from src.core.justify import justify
from src.core.remember import rank_rows


def run(spec, rows):
    recommendation = rank_rows(spec, rows, top_k=1)[0][1]
    return {"recommendation": recommendation, **justify(spec, rows, recommendation)}
