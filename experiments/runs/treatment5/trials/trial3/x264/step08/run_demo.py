import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from fix_configuration import main

dataset = Path(sys.argv[1])
print(json.dumps(main(dataset), indent=2, sort_keys=True))
