import csv
import json
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

step = Path(__file__).resolve().parent
spec = spec_from_file_location("summarize_x264", step / "source" / "summarize_x264.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
result = module.compare_search_strategies(Path(sys.argv[1]), budget=20, seeds=range(42, 52), tie_tolerance=1e-12)
print(json.dumps(result, indent=2, sort_keys=True))
with (step / "metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc", "test_pass", "runtime_seconds", "error_count", "notes"])
    writer.writeheader()
    writer.writerow({"step": 10, "treatment": 4, "trial": 3, "domain": "video encoding", "skill": "race", "total_loc": "", "new_loc": "", "test_pass": "", "runtime_seconds": "", "error_count": "", "notes": "populated after verification"})
