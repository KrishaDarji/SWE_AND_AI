"""Command-line entry point for the reusable project-history summarizer."""

import argparse

from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NASA93 historical delivery outcomes")
    parser.add_argument("csv_path")
    args = parser.parse_args()
    summary = summarize_projects(CsvProjectRepository(args.csv_path).projects())
    print("\n".join(summary.as_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
