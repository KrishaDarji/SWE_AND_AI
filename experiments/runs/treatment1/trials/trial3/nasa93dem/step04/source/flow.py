#!/usr/bin/env python3
"""Assess project proposals as they arrive during portfolio planning."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

OUTCOMES = ("Kloc+", "Effort-", "Defects-", "MonthsX")
NUMERIC = {"centerX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}


def _value(row, key):
    value = row.get(key, "")
    return value.strip() if isinstance(value, str) else value


def _distance(a, b, fields):
    total = 0.0
    for field in fields:
        av, bv = _value(a, field), _value(b, field)
        if field in NUMERIC:
            try:
                total += abs(float(av) - float(bv)) / 100.0
            except (TypeError, ValueError):
                total += 1.0
        else:
            total += 0.0 if av == bv else 1.0
    return math.sqrt(total)


class ProposalFlow:
    """A streaming, deterministic assessor backed by historical projects."""

    def __init__(self, history):
        if not history:
            raise ValueError("history must contain at least one project")
        self.history = list(history)
        self.fields = [k for k in self.history[0] if k not in OUTCOMES and k != "idX"]

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="", encoding="utf-8") as handle:
            return cls(list(csv.DictReader(handle)))

    def assess(self, proposal, neighbors=5):
        ranked = sorted(((_distance(proposal, row, self.fields), row) for row in self.history), key=lambda x: x[0])
        selected = ranked[:max(1, min(neighbors, len(ranked)))]
        estimates = {}
        for outcome in OUTCOMES:
            values = [float(_value(row, outcome)) for _, row in selected]
            estimates[outcome] = round(sum(values) / len(values), 3)
        distance = selected[-1][0]
        flags = []
        if distance > 3.0:
            flags.append("low_similarity")
        if estimates["MonthsX"] > 18:
            flags.append("long_schedule")
        if estimates["Defects-"] > 500:
            flags.append("high_defect_exposure")
        return {
            "proposal": proposal.get("id", proposal.get("name", "unnamed")),
            "estimates": estimates,
            "neighbors": len(selected),
            "similarity_distance": round(distance, 4),
            "confidence": "high" if distance < 1.5 else "medium" if distance < 3.0 else "low",
            "risk_flags": flags,
        }

    def assess_stream(self, proposals):
        for proposal in proposals:
            result = self.assess(proposal)
            yield result
            # Accepted proposals become context for later arrivals when outcomes exist.
            if all(outcome in proposal for outcome in OUTCOMES):
                self.history.append(proposal)


def read_proposals(path):
    if path == "-":
        return [json.loads(line) for line in sys.stdin if line.strip()]
    with open(path, encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".json":
            data = json.load(handle)
            return data if isinstance(data, list) else [data]
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--proposals", required=True, help="JSON/JSONL stdin (-), JSON, or CSV")
    args = parser.parse_args(argv)
    flow = ProposalFlow.from_csv(args.dataset)
    for result in flow.assess_stream(read_proposals(args.proposals)):
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
