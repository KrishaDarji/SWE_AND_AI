#!/usr/bin/env python3
import argparse,json
from analyzer import flow_dataset

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Stream validation results in CSV order.")
    p.add_argument("dataset"); a=p.parse_args()
    print(json.dumps(flow_dataset(a.dataset),indent=2,sort_keys=True,allow_nan=False))
