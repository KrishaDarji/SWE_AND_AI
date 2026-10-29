import csv
import json
import sys
from pathlib import Path

source = Path(__file__).parent / "source"
sys.path.insert(0, str(source))
import summarize_x264 as implementation

dataset = Path(__file__).parents[7] / "moot/optimize/config/X264_AllMeasurements.csv"
with dataset.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
settings = [field for field in rows[0] if field != implementation.OUTCOME]
outcomes = {}
for row in rows:
    key = implementation._configuration_key({field: row[field] for field in settings})
    outcomes[key] = float(row[implementation.OUTCOME])
candidates = [{field: row[field] for field in settings} for row in rows]
result = implementation.choose_configuration(
    candidates, lambda configuration: outcomes[implementation._configuration_key(configuration)], 20, 42)
result["audit"] = implementation.audit_observed_minimum(rows)
print(json.dumps(result, indent=2, sort_keys=True))
