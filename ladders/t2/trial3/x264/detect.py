#!/usr/bin/env python3
import argparse
import sys
from x264_summary.anomaly import detect_anomalies


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect anomalous benchmark measurements")
    parser.add_argument("csv_path")
    parser.add_argument("--measure", default="")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args(argv)
    try:
        for item in detect_anomalies(args.csv_path, args.measure, args.threshold):
            settings = ",".join(f"{k}={v}" for k, v in item.configuration.items())
            print(f"row={item.row_number} {settings} {item.measure}={item.value:.6f} "
                  f"score={item.score:.6f} reason={item.reason}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
