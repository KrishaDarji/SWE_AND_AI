import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import summary

dataset = sys.argv[1]
print("SUMMARY")
print(json.dumps(summary.summarize(dataset), sort_keys=True))
print("BLAME")
print(json.dumps(summary.blame(dataset), sort_keys=True))
print("CERTIFY")
print(json.dumps(summary.certify(dataset), sort_keys=True))
print("FLOW")
print(json.dumps(summary.flow_final(dataset), sort_keys=True))
print("PREDICTION")
print(json.dumps({k: v for k, v in summary.evaluate(dataset).items() if k != "model"}, sort_keys=True))
