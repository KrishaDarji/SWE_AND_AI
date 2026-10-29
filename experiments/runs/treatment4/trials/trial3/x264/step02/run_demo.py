import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
import summarize_x264
print(json.dumps({k: v for k, v in summarize_x264.evaluate(sys.argv[1]).items() if k not in {"model", "settings", "test_groups", "training_frame"}}, indent=2, sort_keys=True))
