import csv, json, sys
from pathlib import Path
from source.nasa93_summary import evaluate, fit_final_model, predict_proposal
path = Path(sys.argv[1]); result = evaluate(path)
print(json.dumps({k: result[k] for k in ('predictors','train_count','test_count','mae','baseline_mae','settings')}, indent=2))
model, predictors = fit_final_model(path)
proposal = {c: None for c in predictors}
print(json.dumps({'final_model': 'fit on all historical rows', 'proposal_prediction': predict_proposal(model, predictors, proposal)}))
