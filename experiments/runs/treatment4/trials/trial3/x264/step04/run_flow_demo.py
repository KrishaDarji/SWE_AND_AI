#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from summarize_x264 import benchmark_csv

snapshots = list(benchmark_csv(sys.argv[1]))
final = snapshots[-1] if snapshots else None
print(json.dumps({"snapshot_count": len(snapshots), "final_snapshot": final}, sort_keys=True))
