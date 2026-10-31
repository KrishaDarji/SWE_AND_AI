import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from flow_estimator import assess, load_csv, write_metrics

root = Path(__file__).parent
records = assess(load_csv(Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")))
write_metrics(records, root / "arrival_results.csv")
print("processed", len(records), "scored arrivals")
print("csv order is a demonstration sequence, not a verified timeline")
