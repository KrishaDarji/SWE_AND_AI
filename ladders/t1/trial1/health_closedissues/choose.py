#!/usr/bin/env python3
"""Choose the best observed model configuration under a metric constraint."""

import argparse
import csv
import sys


def choose_configuration(path, min_accuracy=None, max_error=None):
    """Return ``(row, accuracy, error)`` for the best eligible row.

    Exactly one constraint is required. Accuracy constraints maximize ACC+;
    error constraints minimize MRE-. Ties are resolved by the other metric,
    then by the original row order for deterministic results.
    """
    if (min_accuracy is None) == (max_error is None):
        raise ValueError("provide exactly one of min_accuracy or max_error")
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"MRE-", "ACC+"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain MRE- and ACC+ columns")
        best = None
        for line_number, row in enumerate(reader, start=2):
            try:
                error = float(row["MRE-"])
                accuracy = float(row["ACC+"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid metric at CSV line %d" % line_number) from exc
            if min_accuracy is not None:
                eligible = accuracy >= min_accuracy
                key = (-accuracy, error, line_number)
            else:
                eligible = error <= max_error
                key = (error, -accuracy, line_number)
            if eligible and (best is None or key < best[0]):
                best = (key, dict(row), accuracy, error)
    if best is None:
        constraint = ("ACC+ >= %.6g" % min_accuracy if min_accuracy is not None
                      else "MRE- <= %.6g" % max_error)
        raise ValueError("no configuration satisfies %s" % constraint)
    return best[1], best[2], best[3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--min-accuracy", type=float)
    group.add_argument("--max-error", type=float)
    args = parser.parse_args(argv)
    if args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1:
        parser.error("--min-accuracy must be between 0 and 1")
    if args.max_error is not None and args.max_error < 0:
        parser.error("--max-error must be non-negative")
    try:
        row, accuracy, error = choose_configuration(
            args.csv_path, args.min_accuracy, args.max_error)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("mre=%.6f" % error)
    print("accuracy=%.6f" % accuracy)
    print("configuration=" + ",".join("%s=%s" % item for item in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
