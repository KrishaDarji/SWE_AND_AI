#!/usr/bin/env python3
import argparse
import sys
from x264_summary.race import compare_searches


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare bounded system and baseline searches")
    parser.add_argument("csv_path")
    parser.add_argument("--measure", default="")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = compare_searches(args.csv_path, args.measure, args.budget, args.seed)
        print(f"objective_measure={result.objective_measure} budget={result.budget}")
        for item in (result.system, result.baseline):
            settings = ",".join(f"{k}={v}" for k, v in item.configuration.items())
            print(f"{item.strategy} objective={item.objective:.6f} evaluations={item.evaluations} configuration={settings}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
