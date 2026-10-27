#!/usr/bin/env python3
"""Compare the project effort estimator with a historical baseline."""

import argparse
import json

from project_summary import compare_estimators, load_projects


def main():
    parser = argparse.ArgumentParser(description="Compare two effort estimators")
    parser.add_argument("csv_path")
    parser.add_argument("--neighbors", type=int, default=7)
    args = parser.parse_args()
    result = compare_estimators(load_projects(args.csv_path), args.neighbors)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
