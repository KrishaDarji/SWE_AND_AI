#!/usr/bin/env python3
"""Certify whether one selected process group is less costly."""
import argparse
import json
from project_summary import certify_less_costly, load_projects


def main():
    parser = argparse.ArgumentParser(description="Compare two process groups by effort")
    parser.add_argument("csv_path")
    parser.add_argument("group_column")
    parser.add_argument("process_a")
    parser.add_argument("process_b")
    parser.add_argument("--cost", default="effort")
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    projects = load_projects(args.csv_path)
    for value in (args.process_a, args.process_b):
        if value not in {p.attributes.get(args.group_column, "") for p in projects}:
            parser.error("process value not found in column %s: %s" % (args.group_column, value))
    first = [p for p in projects if p.attributes.get(args.group_column) == args.process_a]
    second = [p for p in projects if p.attributes.get(args.group_column) == args.process_b]
    result = certify_less_costly(first, second, args.process_a, args.process_b,
                                 args.cost, args.confidence, args.resamples)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
