#!/usr/bin/env python3
"""CLI for estimating effort before project approval."""

import argparse
import json
from project_summary import estimate_effort, load_projects


def main():
    parser = argparse.ArgumentParser(description="Estimate proposed project effort")
    parser.add_argument("csv_path")
    parser.add_argument("proposed_kloc", type=float)
    parser.add_argument("--neighbors", type=int, default=7)
    args = parser.parse_args()
    result = estimate_effort(load_projects(args.csv_path), args.proposed_kloc,
                             args.neighbors)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
