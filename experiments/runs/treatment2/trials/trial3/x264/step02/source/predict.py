#!/usr/bin/env python3
import argparse
import sys
from x264_summary.predict import predict_encoding_time


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict x264 encoding time")
    parser.add_argument("csv_path")
    parser.add_argument("settings", nargs="+", metavar="NAME=VALUE")
    parser.add_argument("--measure", default="")
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        configuration = dict(item.split("=", 1) for item in args.settings)
        if any("=" not in item for item in args.settings):
            raise ValueError("settings must use NAME=VALUE")
        result = predict_encoding_time(args.csv_path, configuration,
                                       args.measure, args.neighbors)
        print(f"predicted_encoding_time={result.value:.6f}")
        print(f"neighbors={result.neighbors} nearest_distance={result.distance:.6f}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
