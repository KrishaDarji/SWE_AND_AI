import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
import summarize_x264
print(json.dumps(summarize_x264.certify(sys.argv[1]), indent=2, sort_keys=True))
