import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import summary
p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--threshold",type=float,default=5); p.add_argument("--mode",choices=["certify","summary","prediction"],default="certify"); a=p.parse_args()
if a.mode == "summary": out=summary.summarize(a.dataset)
elif a.mode == "prediction":
    x=summary.evaluate(a.dataset); out={k:v for k,v in x.items() if k != "model"}
else: out=summary.certify(a.dataset,a.threshold)
print(json.dumps(out, indent=2, sort_keys=True, default=str))
