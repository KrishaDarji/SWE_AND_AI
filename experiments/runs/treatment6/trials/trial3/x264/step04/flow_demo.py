import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
import summary

if __name__ == "__main__":
    print(json.dumps(summary.flow_final(sys.argv[1]), indent=2, sort_keys=True))
