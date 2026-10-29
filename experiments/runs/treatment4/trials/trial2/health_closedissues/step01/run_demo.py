import json
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / "source"))
from summarizer import summarize_csv

data = summarize_csv(Path(__file__).parents[7] / "moot/optimize/hpo/Health-ClosedIssues0000.csv")
print(json.dumps(data, indent=2))
