import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from certify import certify

root = Path(__file__).parent
dataset = root.parents[6] / "moot" / "optimize" / "hpo" / "Health-ClosedIssues0000.csv"
print(json.dumps(certify(dataset), indent=2, sort_keys=True))
