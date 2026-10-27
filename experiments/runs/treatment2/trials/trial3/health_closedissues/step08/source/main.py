"""Command-line entry point for model summary, prediction, and certification."""
import argparse
import json
from model_summary import (summarize_csv, predict_quality, certify_families,
                           evaluate_stream, blame_high_error, justify_configuration,
                           choose_configuration, suggest_smallest_change)

def main():
    parser = argparse.ArgumentParser(description="Summarize, predict, or certify model configurations")
    parser.add_argument("csv_path")
    parser.add_argument("--quality-column", action="append", dest="quality_columns")
    parser.add_argument("--predict-config")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--family-column", action="append", dest="family_columns")
    parser.add_argument("--evaluate-stream", action="store_true")
    parser.add_argument("--blame", action="store_true")
    parser.add_argument("--error-column", default="MRE-")
    parser.add_argument("--high-percentile", type=float, default=0.75)
    parser.add_argument("--justify", action="store_true")
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--suggest-smallest-change", action="store_true")
    parser.add_argument("--prediction-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.suggest_smallest_change:
        result = suggest_smallest_change(args.csv_path, args.error_column)
    elif args.choose:
        configuration = json.loads(args.predict_config) if args.predict_config else None
        result = choose_configuration(args.csv_path, configuration, args.quality_columns,
                                      args.neighbors, args.prediction_weight)
    elif args.justify:
        configuration = json.loads(args.predict_config) if args.predict_config else None
        result = justify_configuration(args.csv_path, configuration, args.quality_columns,
                                       args.error_column, args.high_percentile)
    elif args.blame:
        result = blame_high_error(args.csv_path, args.error_column, args.high_percentile)
    elif args.evaluate_stream:
        records = (json.loads(line) for line in __import__("sys").stdin if line.strip())
        result = evaluate_stream(records, args.quality_columns)
    elif args.predict_config:
        result = predict_quality(args.csv_path, json.loads(args.predict_config), args.quality_columns, args.neighbors)
    elif args.family_columns:
        result = certify_families(args.csv_path, args.family_columns, args.quality_columns)
    else:
        result = summarize_csv(args.csv_path, quality_columns=args.quality_columns)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
