import csv
import json
from pathlib import Path
from source.choose import select, audit

BASE = Path(__file__).resolve().parent
DATA = Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")

def load():
    with DATA.open(newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    value_at = header.index("Performance-")
    candidates = [tuple(int(x) for x in row[:value_at]) for row in rows[1:]]
    values = [float(row[value_at]) for row in rows[1:]]
    return candidates, values

def main():
    candidates, values = load()
    calls = []
    def evaluate(config):
        index = candidates.index(config)
        calls.append(index)
        return values[index]
    result = select(candidates, evaluate, budget=20, seed=42)
    measured = {int(identifier[1:]) for identifier, _, _ in result.observations}
    audit_row = audit(candidates, lambda config: values[candidates.index(config)])
    gap = result.observations[[x[0] for x in result.observations].index(result.recommended_id)][2] - audit_row[2]
    print(json.dumps({"evaluation_order": result.evaluation_order,
                      "observations": result.observations,
                      "budget_used": result.budget_used,
                      "recommended_id": result.recommended_id,
                      "recommended_configuration": result.recommended_configuration,
                      "audit_observed_minimum": audit_row[2], "audit_gap": gap,
                      "callback_calls": calls, "audited_after_freeze": True}, default=list))

if __name__ == "__main__":
    main()
