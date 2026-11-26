"""Step 1 (remember): summarize the historical data and show the best rows."""

from src.core.datasets import features
from src.core.remember import rank_rows, summarize


def run(spec, rows):
    return {
        "rows": len(rows),
        "features": features(spec, rows),
        "objectives": summarize(spec, rows=rows)["objectives"],
        "best": [{"score": score, "row": row} for score, row in rank_rows(spec, rows, top_k=3)],
    }
