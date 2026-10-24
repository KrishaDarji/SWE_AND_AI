"""Run the human/shared ten-step ladder across all selected domains."""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.blame import blame
from src.core.certify import certify
from src.core.choose import choose
from src.core.datasets import DATASETS, read_rows
from src.core.fix import fix
from src.core.flow import stream_objective
from src.core.guess import leave_one_out_mae
from src.core.justify import justify
from src.core.race import race
from src.core.remember import rank_rows, summarize
from src.core.spot import spot


OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def run_domain(name: str, max_rows: int = 300):
    spec = DATASETS[name]
    rows = read_rows(spec, limit=max_rows)
    first = rows[0]
    ranked = rank_rows(spec, rows, top_k=1)
    candidate = ranked[0][1]
    steps = []

    def record(step, payload):
        steps.append({"step": step, "payload": payload})

    record(1, summarize(spec, limit=max_rows))
    record(2, {"leave_one_out_mae": leave_one_out_mae(spec, rows, k=5)})
    for objective in spec.objectives:
        values = [float(row[objective]) for row in rows if row.get(objective) not in (None, "", "?")]
        midpoint = len(values) // 2
        record(3, {objective: certify(objective, values[:midpoint], values[midpoint:], permutations=500)})
    record(4, stream_objective(spec, rows))
    record(5, {"findings": blame(spec, rows, top_k=5)})
    record(6, justify(spec, rows, candidate, top_k=3))
    selected = choose(spec, rows, budget=min(20, len(rows)))
    record(7, selected)
    record(8, fix(spec, rows, selected["selected"]))
    record(9, {"findings": spot(spec, rows, top_k=5)})
    record(10, race(spec, rows, budget=min(20, len(rows))))
    return {"domain": name, "rows_used": len(rows), "steps": steps}


def main():
    results = [run_domain(name) for name in DATASETS]
    output = OUT / "treatment7_ladder.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = OUT / "treatment7_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["domain", "step", "payload_keys"])
        for result in results:
            for step in result["steps"]:
                writer.writerow([result["domain"], step["step"], ";".join(sorted(step["payload"]))])
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
