#!/usr/bin/env python3
import argparse, json
from analyzer import race

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Compare historical configuration selection with uniform baseline sampling.")
    p.add_argument("dataset"); p.add_argument("--cap",type=float,default=100); p.add_argument("--tie-tolerance",type=float,default=1e-12)
    a=p.parse_args(); print(json.dumps(race(a.dataset,a.cap,range(42,52),a.tie_tolerance),indent=2,sort_keys=True,allow_nan=False))
