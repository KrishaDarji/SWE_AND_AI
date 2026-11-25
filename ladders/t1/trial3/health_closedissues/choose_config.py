#!/usr/bin/env python3
"""Choose a balanced model configuration from evaluated CSV candidates."""
import argparse, csv, json, sys
from pathlib import Path

REQUIRED = ("MRE-", "ACC+", "PRED40+")
PARAMETERS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

def choose(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(k not in reader.fieldnames for k in REQUIRED):
            raise ValueError("CSV must contain MRE-, ACC+, and PRED40+ columns")
        best = None
        for index, row in enumerate(reader):
            values = []
            for name in REQUIRED:
                try:
                    value = float(row[name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("invalid {} on CSV line {}".format(name, index + 2)) from exc
                if not 0.0 <= value <= 1.0:
                    raise ValueError("{} on CSV line {} is outside [0, 1]".format(name, index + 2))
                values.append(value)
            mre, acc, pred = values
            score = 0.2 * (1.0 - mre) + 0.4 * acc + 0.4 * pred
            key = (score, -mre, acc, pred, -index)
            if best is None or key > best[0]:
                best = (key, row, score, index)
    if best is None:
        raise ValueError("CSV contains no candidate rows")
    _, row, score, index = best
    return {"row_number": index + 2, "score": score,
            "metrics": {k: float(row[k]) for k in REQUIRED},
            "configuration": {k: row[k] for k in PARAMETERS if k in row}}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    try:
        result = choose(parser.parse_args(argv).csv_path)
    except (OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
