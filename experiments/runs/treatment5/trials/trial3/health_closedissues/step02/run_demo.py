import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from guess_model import fit_csv
path = "/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv"
model = fit_csv(path)
print(json.dumps({"predictors": model.schema, "train_rows": len(model.train), "test_rows": len(model.test),
                  "train_groups": len(model.train_keys), "test_groups": len(model.test_keys),
                  "split": "seed=42; shuffle normalized groups; approximately 80/20", "metrics": model.evaluate()}, sort_keys=True))
print(json.dumps({"N_estimators": 100, "criterion": "squared_error", "Min_sample_leaves": 2,
                  "Min_impurity_decrease": 1.0, "Max_depth": 8, "prediction": model.predict({"N_estimators": 100,
                  "criterion": "squared_error", "Min_sample_leaves": 2, "Min_impurity_decrease": 1.0, "Max_depth": 8})}, sort_keys=True))
