import sys, json
sys.path.insert(0, "source")
from summarize_runs import evaluate, predict_configuration, render, analyze
dataset = sys.argv[1]
result = evaluate(dataset)
print(json.dumps({"predictors": result["predictors"], "rows": [result["train_rows"], result["test_rows"]], "groups": [result["train_groups"], result["test_groups"]], "targets": {k:{x:v[x] for x in ("finite_training","finite_test","mae","baseline_mae","training_mean")} for k,v in result["targets"].items()}, "prediction": predict_configuration(result, {"N_estimators":40,"criterion":"squared_error","Min_sample_leaves":3,"Min_impurity_decrease":2.75,"Max_depth":10})}, indent=2, sort_keys=True))
print(render(analyze(dataset)))
