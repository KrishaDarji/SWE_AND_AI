import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import summary

path = sys.argv[1]
result = summary.evaluate(path)
print(json.dumps({key: value for key, value in result.items() if key not in ("model", "train_indexes", "test_indexes")}, indent=2, sort_keys=True))
settings = result["settings"]
names, rows, _ = summary.read_runs(path)
print(json.dumps({"supplied_configuration_prediction": result["model"].predict_one({name: rows[0][name] for name in settings}),
                  "configuration_fields": settings}, sort_keys=True))
