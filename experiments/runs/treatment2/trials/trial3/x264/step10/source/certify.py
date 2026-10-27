#!/usr/bin/env python3
import argparse
import sys
from x264_summary.certify import certify_improvement


def _config(items):
    if any("=" not in item for item in items):
        raise ValueError("settings must use NAME=VALUE")
    return dict(item.split("=", 1) for item in items)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Certify a configuration improvement")
    parser.add_argument("csv_path")
    parser.add_argument("--baseline", nargs="+", required=True, metavar="NAME=VALUE")
    parser.add_argument("--candidate", nargs="+", required=True, metavar="NAME=VALUE")
    parser.add_argument("--measure", default="")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = certify_improvement(args.csv_path, _config(args.baseline), _config(args.candidate),
                                     args.measure, args.alpha, args.higher_is_better)
        for name in ("baseline_n", "candidate_n", "baseline_mean", "candidate_mean",
                     "improvement", "t_statistic", "p_value", "confidence_low",
                     "confidence_high", "credible"):
            print(f"{name}={getattr(result, name)}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
