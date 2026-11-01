#!/usr/bin/env python3
import argparse,json
from certify import certify
p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--threshold",type=float); a=p.parse_args(); print(json.dumps(certify(a.dataset,a.threshold),indent=2,sort_keys=True,allow_nan=False))
