import argparse, json
from predictor import read_csv, train_evaluate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("dataset"); args=ap.parse_args()
    names, data=read_csv(args.dataset); model, metrics=train_evaluate(data,names)
    metrics.update(predictors=names, model="standardized ridge regression", seed=42, split="80/20 configuration-group split")
    print(json.dumps(metrics, sort_keys=True))
if __name__ == "__main__": main()
