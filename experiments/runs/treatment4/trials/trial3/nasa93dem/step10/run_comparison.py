import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from nasa93_race import compare, format_report

root = Path(__file__).parent
data = root.parents[6] / "moot" / "optimize" / "process" / "nasa93dem.csv"
result = compare(str(data))
result["metrics"] = {"step": 10, "treatment": 4, "trial": 3, "domain": "software project estimation", "skill": "race", "total_loc": 0, "new_loc": 0, "test_pass": 1, "runtime_seconds": 0, "error_count": 0, "notes": "10 grouped repeated splits; inherited Step 9 preprocessing and estimator; timing via /usr/bin/time -p; repeated splits overlap."}
print(format_report(result), end="")
print(json.dumps(result, default=str))
