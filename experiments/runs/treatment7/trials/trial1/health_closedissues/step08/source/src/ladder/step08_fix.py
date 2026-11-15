"""Step 8 (fix): smallest (one-setting) change that improves the worst row."""

from src.core.fix import fix
from src.core.remember import rank_rows


def run(spec, rows):
    worst = rank_rows(spec, rows, top_k=len(rows))[-1][1]
    return {"poor_candidate": worst, **fix(spec, rows, worst)}
