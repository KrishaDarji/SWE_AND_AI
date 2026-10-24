"""Run Treatment 7, Step 1 for one selected domain."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import get_dataset, read_rows
from src.core.remember import rank_rows, summarize


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--top", type=int, default=5)
args = parser.parse_args()

spec = get_dataset(args.dataset)
rows = read_rows(spec)
ranked = rank_rows(spec, rows, top_k=args.top)

print(json.dumps({"summary": summarize(spec), "top_rows": [{"score": score, "row": row} for score, row in ranked]}, indent=2))
