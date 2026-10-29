import json
from pathlib import Path
import sys
root = Path(__file__).parent
sys.path.insert(0, str(root / "source"))
from predictor import evaluate, fit_final, predict_configuration
dataset = root.parents[6] / "moot" / "optimize" / "hpo" / "Health-ClosedIssues0000.csv"
result = evaluate(dataset)
print(json.dumps({k: v for k, v in result.items() if k in ("predictor_names", "train_count", "test_count", "metrics", "training_mean_baselines", "estimator_settings")}, sort_keys=True))
models = fit_final(dataset)
print(json.dumps(predict_configuration(models, {"N_estimators": 100, "criterion": "squared_error", "Min_sample_leaves": 2, "Min_impurity_decrease": 0.5, "Max_depth": 10}), sort_keys=True))
