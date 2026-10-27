#!/usr/bin/env python3
import argparse
import sys
from x264_summary.justify import justify_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description="Justify an x264 configuration")
    parser.add_argument("csv_path")
    parser.add_argument("settings", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--measure", default="")
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    try:
        if any("=" not in item for item in args.settings):
            raise ValueError("settings must use NAME=VALUE")
        config = dict(item.split("=", 1) for item in args.settings) or None
        result = justify_configuration(args.csv_path, config, args.measure, args.higher_is_better)
        print("configuration=" + ",".join(f"{k}={v}" for k, v in result.configuration.items()))
        print(f"measure={result.measure} samples={result.samples} mean={result.mean:.6f}")
        if result.margin is not None:
            print(f"runner_up_mean={result.runner_up_mean:.6f} margin={result.margin:.6f}")
        print("justification=" + result.recommendation)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
