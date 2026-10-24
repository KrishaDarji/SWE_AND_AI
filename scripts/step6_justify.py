"""Run Treatment 7, Step 6: explain the best observed recommendation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import get_dataset, read_rows
from src.core.justify import justify
from src.core.remember import rank_rows


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--top", type=int, default=3)
args = parser.parse_args()

spec = get_dataset(args.dataset)
rows = read_rows(spec)
recommendation = rank_rows(spec, rows, top_k=1)[0][1]
print(json.dumps(justify(spec, rows, recommendation, top_k=args.top), indent=2))
