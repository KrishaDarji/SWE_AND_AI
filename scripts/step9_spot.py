"""Run Treatment 7, Step 9: find anomalous records."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import get_dataset, read_rows
from src.core.spot import spot


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--top", type=int, default=10)
args = parser.parse_args()

spec = get_dataset(args.dataset)
print(json.dumps(spot(spec, read_rows(spec), top_k=args.top), indent=2))
