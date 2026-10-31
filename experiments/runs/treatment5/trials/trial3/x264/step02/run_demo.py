import os, sys, json
root=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(root,"source"))
from predictor import train_evaluate
data="/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv"
r=train_evaluate(data)
print(json.dumps({k:v for k,v in r.items() if k != "model"}, indent=2))
print("sample_prediction", r["model"].predict_one(dict(zip(r["predictor_names"], [1]*len(r["predictor_names"])))) )
