#!/usr/bin/env python3
"""CLI for spotting a project record inconsistent with historical effort rates."""
import argparse
import json

from project_summary import load_projects, spot_inconsistent


def main():
    parser = argparse.ArgumentParser(description="Spot an inconsistent historical project")
    parser.add_argument("csv_path")
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps([item.to_dict() for item in spot_inconsistent(
        load_projects(args.csv_path), args.top)], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
