import argparse, csv, json
from estimator import read_csv, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--proposal", help="JSON object containing exactly the predictor fields")
    parser.add_argument("--metrics", help="CSV output path")
    args = parser.parse_args()
    result = evaluate(read_csv(args.dataset))
    print("predictors=" + ",".join(result["predictors"]))
    print("train_count=%d test_count=%d" % (len(result["train"]), len(result["test"])))
    print("split=grouped exact predictor configurations, seed=42, test_fraction=0.25")
    print("model=categorical prefix backoff group means, smoothing=1.0, missing=<MISSING>")
    print("held_out_mae=%.6f baseline_mae=%.6f" % (result["mae"], result["baseline_mae"]))
    if args.proposal:
        proposal = json.loads(args.proposal)
        print("proposal_prediction=%.6f" % result["model"].predict_one(proposal))
    if args.metrics:
        with open(args.metrics, "w", newline="") as handle:
            csv.writer(handle).writerows([["measure", "value"], ["held_out_mae", result["mae"]], ["baseline_mae", result["baseline_mae"]]])
if __name__ == "__main__":
    main()
