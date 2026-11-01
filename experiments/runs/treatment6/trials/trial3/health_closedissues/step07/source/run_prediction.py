#!/usr/bin/env python3
import argparse, json
from analyzer import predict_dataset
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    print(json.dumps(predict_dataset(a.dataset,a.seed),indent=2,sort_keys=True,allow_nan=False))
