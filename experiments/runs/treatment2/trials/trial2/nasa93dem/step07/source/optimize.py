#!/usr/bin/env python3
"""CLI for choosing project settings under a fixed effort budget."""
import argparse
import json
from project_summary import choose_settings, load_projects


def main():
    parser = argparse.ArgumentParser(description="Choose settings maximizing delivered value under budget")
    parser.add_argument("csv_path")
    parser.add_argument("budget", type=float)
    parser.add_argument("--settings", nargs="+", help="categorical CSV columns to group by")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--minimum-group-size", type=int, default=2)
    parser.add_argument("--value", choices=("kloc", "defects", "months"), default="kloc")
    args = parser.parse_args()
    result = choose_settings(load_projects(args.csv_path), args.budget, args.settings,
                             args.top, args.minimum_group_size, args.value)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
