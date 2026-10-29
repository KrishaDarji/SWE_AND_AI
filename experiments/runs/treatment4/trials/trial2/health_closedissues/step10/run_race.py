import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from race import compare

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
print(json.dumps(compare(args.dataset), indent=2, sort_keys=True))
