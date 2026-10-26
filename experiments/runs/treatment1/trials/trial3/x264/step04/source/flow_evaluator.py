#!/usr/bin/env python3
"""One-at-a-time benchmark result evaluator for x264 measurements."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TextIO


@dataclass
class Best:
    score: float
    configuration: Dict[str, str]
    index: int


class FlowEvaluator:
    """Consume benchmark rows in arrival order and retain the best result."""

    def __init__(self, score_field: str):
        self.score_field = score_field
        self.count = 0
        self.best: Optional[Best] = None

    def evaluate(self, row: Dict[str, str]) -> Dict[str, object]:
        self.count += 1
        try:
            score = float(row[self.score_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("row has no numeric %s" % self.score_field) from exc
        if self.best is None or score < self.best.score:
            self.best = Best(score, dict(row), self.count)
            improved = True
        else:
            improved = False
        return self.snapshot(improved)

    def snapshot(self, improved: bool = False) -> Dict[str, object]:
        if self.best is None:
            return {"seen": 0, "improved": False, "best": None}
        return {
            "seen": self.count,
            "improved": improved,
            "best": {
                "score": self.best.score,
                "index": self.best.index,
                "configuration": self.best.configuration,
            },
        }


def rows_from_csv(handle: TextIO) -> Iterable[Dict[str, str]]:
    reader = csv.DictReader(handle, skipinitialspace=True)
    if not reader.fieldnames or len(reader.fieldnames) < 2:
        raise ValueError("CSV must contain configuration fields and a score field")
    for row in reader:
        if any(value is None for value in row.values()):
            raise ValueError("CSV row has the wrong number of columns")
        yield row


def run(handle: TextIO, output: TextIO, score_field: str) -> int:
    evaluator = FlowEvaluator(score_field)
    for row in rows_from_csv(handle):
        output.write(json.dumps(evaluator.evaluate(row), sort_keys=True) + "\n")
    output.write(json.dumps(evaluator.snapshot(), sort_keys=True) + "\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", help="CSV path; stdin when omitted")
    parser.add_argument("--score-field", default="Performance-", dest="score_field")
    args = parser.parse_args(argv)
    if args.csv_path:
        with open(args.csv_path, newline="") as handle:
            return run(handle, sys.stdout, args.score_field)
    return run(sys.stdin, sys.stdout, args.score_field)


if __name__ == "__main__":
    raise SystemExit(main())
