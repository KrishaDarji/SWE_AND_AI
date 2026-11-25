#!/usr/bin/env python3
import argparse
import sys
from x264_summary.core import render, summarize_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize encoder settings and measures")
    parser.add_argument("csv_path")
    parser.add_argument("--measure", action="append", default=[],
                        help="measure column (repeatable; default: final column)")
    args = parser.parse_args(argv)
    try:
        print(render(summarize_csv(args.csv_path, args.measure)), end="")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
