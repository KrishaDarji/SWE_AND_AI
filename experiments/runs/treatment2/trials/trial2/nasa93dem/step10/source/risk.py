#!/usr/bin/env python3
"""CLI for recommending a low-cost process change that reduces risk."""
import argparse
import json
from project_summary import load_projects, recommend_process_change


def main():
    parser = argparse.ArgumentParser(description="Recommend a cheaper risk-reducing process change")
    parser.add_argument("csv_path")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--minimum-group-size", type=int, default=3)
    args = parser.parse_args()
    result = recommend_process_change(load_projects(args.csv_path), args.top,
                                      args.minimum_group_size)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
