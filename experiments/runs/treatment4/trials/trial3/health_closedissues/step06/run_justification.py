import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "source"))
from justify import justify

print(json.dumps(justify(sys.argv[1]), indent=2, sort_keys=True))
