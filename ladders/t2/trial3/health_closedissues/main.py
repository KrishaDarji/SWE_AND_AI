"""Command-line entry point for the reusable model summary."""

import argparse
import json
from model_summary import summarize_csv


def main():
    parser = argparse.ArgumentParser(description="Summarize model quality and configuration CSV columns")
    parser.add_argument("csv_path")
    parser.add_argument("--quality-column", action="append", dest="quality_columns")
    args = parser.parse_args()
    summary = summarize_csv(args.csv_path, quality_columns=args.quality_columns)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
