import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from summarize_x264 import repair

print(json.dumps(repair(Path(sys.argv[1])), indent=2, sort_keys=True))
