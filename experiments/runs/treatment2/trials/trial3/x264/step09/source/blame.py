#!/usr/bin/env python3
import argparse
import sys
from x264_summary.blame import blame_slow_encodes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explain settings associated with slow encodes")
    parser.add_argument("csv_path")
    parser.add_argument("--measure", default="")
    args = parser.parse_args(argv)
    try:
        for item in blame_slow_encodes(args.csv_path, args.measure):
            print(f"setting={item.setting} enabled_n={item.enabled_n} disabled_n={item.disabled_n} "
                  f"enabled_mean={item.enabled_mean:.6f} disabled_mean={item.disabled_mean:.6f} "
                  f"penalty={item.penalty:.6f}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
