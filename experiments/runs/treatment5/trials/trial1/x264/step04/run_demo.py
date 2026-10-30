import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from flow import evaluate, csv_records

snapshots = evaluate(csv_records(sys.argv[1]))
print("processed_snapshots=" + str(len(snapshots)))
print(snapshots[-1] if snapshots else {"count": 0})
