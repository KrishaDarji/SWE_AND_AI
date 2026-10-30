import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent / 'source'))
from repair import read_csv, select
import json
print(json.dumps(select(read_csv(sys.argv[1])), indent=2, sort_keys=True))
