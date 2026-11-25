#!/usr/bin/env python3
"""CLI for the reusable historical project summary."""

import argparse
import json
from project_summary import load_projects, summarize_projects


def main():
    parser = argparse.ArgumentParser(description="Summarize NASA historical projects")
    parser.add_argument("csv_path")
    args = parser.parse_args()
    projects = load_projects(args.csv_path)
    result = summarize_projects(projects, args.csv_path)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
