#!/usr/bin/env python3
import argparse
import csv
import sys
from x264_summary.stream import StreamingEvaluator


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate benchmark rows incrementally")
    parser.add_argument("csv_path")
    parser.add_argument("--measure", default="")
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    try:
        with open(args.csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV must contain a header")
            measure = args.measure or reader.fieldnames[-1]
            if measure not in reader.fieldnames:
                raise ValueError("unknown measure column: " + measure)
            evaluator = StreamingEvaluator([n for n in reader.fieldnames if n != measure],
                                            measure, args.higher_is_better)
            for row in reader:
                evaluator.add(row)
        best = evaluator.best_configuration()
        if best is None:
            raise ValueError("CSV must contain at least one data row")
        print("best=" + ",".join(f"{n}={v}" for n, v in zip(evaluator.setting_columns, best)))
        for result in evaluator.results():
            print(f"samples={result.samples} mean={result.mean:.6f} best={result.best} "
                  + ",".join(f"{k}={v}" for k, v in result.configuration.items()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
