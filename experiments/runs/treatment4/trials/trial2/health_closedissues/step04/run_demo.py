import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from flow import running_summaries_csv

result = running_summaries_csv(sys.argv[1])
print(json.dumps({"snapshots": len(result), "final": result[-1] if result else None}, sort_keys=True))
