"""Run Treatment 7, Step 2: guess objective values."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import get_dataset, read_rows
from src.core.guess import leave_one_out_mae, predict_row


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

spec = get_dataset(args.dataset)
rows = read_rows(spec, limit=args.limit)
result = {
    "dataset": spec.name,
    "rows": len(rows),
    "k": args.k,
    "leave_one_out_mae": leave_one_out_mae(spec, rows, k=args.k),
    "first_row_prediction": predict_row(spec, rows[1:], rows[0], k=args.k),
}
print(json.dumps(result, indent=2))
