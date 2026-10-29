import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from summarize_runs import budget_select_dataset

print(json.dumps(budget_select_dataset(Path(sys.argv[1]), 20, 42), indent=2, sort_keys=True))
