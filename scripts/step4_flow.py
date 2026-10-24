"""Run Treatment 7, Step 4: stream rows until estimates stabilize."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.datasets import get_dataset, read_rows
from src.core.flow import stream_objective


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--minimum", type=int, default=20)
parser.add_argument("--tolerance", type=float, default=0.01)
parser.add_argument("--patience", type=int, default=5)
args = parser.parse_args()

spec = get_dataset(args.dataset)
result = stream_objective(spec, read_rows(spec), args.minimum, args.tolerance, args.patience)
print(json.dumps(result, indent=2))
