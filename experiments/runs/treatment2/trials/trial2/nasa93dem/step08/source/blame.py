#!/usr/bin/env python3
"""Explain which project factors are associated with excessive effort."""

import argparse
import json
from project_summary import blame_factors, load_projects


def main():
    parser = argparse.ArgumentParser(description="Rank factors associated with excessive effort")
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--minimum-group-size", type=int, default=3)
    args = parser.parse_args()
    result = blame_factors(load_projects(args.csv_path), args.threshold, args.top,
                           args.minimum_group_size)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
