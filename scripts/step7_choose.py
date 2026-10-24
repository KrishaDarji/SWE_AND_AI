"""Run Treatment 7, Step 7: choose with a restricted evaluation budget."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.choose import choose
from src.core.datasets import get_dataset, read_rows


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["x264", "nasa93dem", "health_closedissues"])
parser.add_argument("--budget", type=int, default=20)
args = parser.parse_args()

spec = get_dataset(args.dataset)
print(json.dumps(choose(spec, read_rows(spec), budget=args.budget), indent=2))
