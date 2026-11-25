#!/usr/bin/env python3
import argparse
import sys
from x264_summary.choose import choose_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description="Choose an x264 configuration under a quality constraint")
    parser.add_argument("csv_path")
    parser.add_argument("--quality-measure", required=True)
    bounds = parser.add_mutually_exclusive_group(required=True)
    bounds.add_argument("--minimum-quality", type=float)
    bounds.add_argument("--maximum-quality", type=float)
    parser.add_argument("--objective-measure", default="")
    parser.add_argument("--quality-lower-is-better", action="store_true")
    parser.add_argument("--objective-higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = choose_configuration(args.csv_path, args.quality_measure,
                                      args.minimum_quality, args.maximum_quality,
                                      args.objective_measure, not args.quality_lower_is_better,
                                      args.objective_higher_is_better)
        print("configuration=" + ",".join(f"{k}={v}" for k, v in result.configuration.items()))
        print(f"quality={result.quality_measure} mean={result.quality_mean:.6f}")
        print(f"objective={result.objective_measure} mean={result.objective_mean:.6f} "
              f"samples={result.samples} feasible={result.feasible_configurations}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
