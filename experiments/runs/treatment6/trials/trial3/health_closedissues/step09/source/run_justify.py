#!/usr/bin/env python3
import argparse, json
from analyzer import justify

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Justify an observed model configuration.'); p.add_argument('dataset'); a=p.parse_args()
    print(json.dumps(justify(a.dataset),indent=2,sort_keys=True,allow_nan=False))
