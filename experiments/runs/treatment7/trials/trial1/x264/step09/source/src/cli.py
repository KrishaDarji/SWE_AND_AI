"""Run ladder steps from the command line.

    python3 -m src.cli x264 3        # one step
    python3 -m src.cli nasa93dem all # all ten steps
"""

import argparse
import json

from src.core.datasets import get_dataset, sample_rows
from src.domains import DATASETS
from src.ladder import run_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("step", help="1-10 or 'all'")
    parser.add_argument("--rows", type=int, default=None, help="sample size (default: adapter max_rows)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    spec = get_dataset(args.dataset)
    rows = sample_rows(spec, args.rows, seed=args.seed)
    steps = range(1, 11) if args.step == "all" else [int(args.step)]
    for step in steps:
        print(json.dumps(run_step(spec, rows, step), indent=2, default=str))


if __name__ == "__main__":
    main()
