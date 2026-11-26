"""Step 4 (flow): read rows one at a time; stop once the estimate is stable."""

from src.core.flow import stream_objective
from src.core.remember import rank_rows, summarize


def run(spec, rows):
    streamed = stream_objective(spec, rows)  # rows arrive in random order
    seen = rows[: streamed["seen"]]
    full = summarize(spec, rows=rows)["objectives"]
    return {
        "seen": streamed["seen"],
        "available": len(rows),
        "stopped_early": streamed["stopped_early"],
        "estimate_at_stop": streamed["final_means"],
        "mean_of_all_rows": {objective: full[objective]["mean"] for objective in full},
        "best_so_far": rank_rows(spec, seen, top_k=1)[0][1],
    }
