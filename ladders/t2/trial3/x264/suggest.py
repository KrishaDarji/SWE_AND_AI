#!/usr/bin/env python3
import argparse
import sys
from x264_summary.suggest import suggest_setting_change


def main(argv=None):
    parser = argparse.ArgumentParser(description="Suggest the smallest x264 setting change")
    parser.add_argument("csv_path")
    parser.add_argument("settings", nargs="+", metavar="NAME=VALUE")
    parser.add_argument("--measure", default="")
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    try:
        if any("=" not in item for item in args.settings):
            raise ValueError("settings must use NAME=VALUE")
        result = suggest_setting_change(args.csv_path, dict(item.split("=", 1) for item in args.settings),
                                        args.measure, args.higher_is_better)
        print("baseline=" + ",".join(f"{k}={v}" for k, v in result.baseline.items()))
        print("candidate=" + ",".join(f"{k}={v}" for k, v in result.candidate.items()))
        print(f"changed_setting={result.setting} baseline_mean={result.baseline_mean:.6f} "
              f"candidate_mean={result.candidate_mean:.6f} improvement={result.improvement:.6f}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
