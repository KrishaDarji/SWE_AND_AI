"""Command-line entry point for model summary, prediction, and certification."""
import argparse
import json
from model_summary import summarize_csv, predict_quality, certify_families

def main():
    parser = argparse.ArgumentParser(description="Summarize, predict, or certify model configurations")
    parser.add_argument("csv_path")
    parser.add_argument("--quality-column", action="append", dest="quality_columns")
    parser.add_argument("--predict-config")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--family-column", action="append", dest="family_columns")
    args = parser.parse_args()
    if args.predict_config:
        result = predict_quality(args.csv_path, json.loads(args.predict_config), args.quality_columns, args.neighbors)
    elif args.family_columns:
        result = certify_families(args.csv_path, args.family_columns, args.quality_columns)
    else:
        result = summarize_csv(args.csv_path, quality_columns=args.quality_columns)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
