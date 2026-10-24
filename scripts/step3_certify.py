"""Run Treatment 7, Step 3: certify an improvement against a baseline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.certify import certify


parser = argparse.ArgumentParser()
parser.add_argument("objective", help="Objective name, e.g. Performance- or ACC+")
parser.add_argument("--treatment", nargs="+", type=float, required=True)
parser.add_argument("--baseline", nargs="+", type=float, required=True)
args = parser.parse_args()

print(json.dumps(certify(args.objective, args.treatment, args.baseline), indent=2))
