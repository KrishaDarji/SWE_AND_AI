"""Command-line entry point for summary and pre-evaluation prediction."""
import argparse
import json
from model_summary import summarize_csv, predict_quality

def main():
    parser = argparse.ArgumentParser(description="Summarize and predict model quality")
    parser.add_argument("csv_path")
    parser.add_argument("--quality-column", action="append", dest="quality_columns")
    parser.add_argument("--predict-config", help="JSON object of configuration settings")
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    if args.predict_config:
        print(json.dumps(predict_quality(args.csv_path, json.loads(args.predict_config),
                                         args.quality_columns, args.neighbors), indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize_csv(args.csv_path, quality_columns=args.quality_columns), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
