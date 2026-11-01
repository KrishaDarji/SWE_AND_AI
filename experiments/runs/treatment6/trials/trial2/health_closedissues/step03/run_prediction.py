import sys, json
sys.path.insert(0, "source")
from summarize_runs import evaluate, predict_configuration
r=evaluate(sys.argv[1])
print(json.dumps({"rows": [r["train_rows"],r["test_rows"]], "groups": [r["train_groups"],r["test_groups"]], "targets": {k:{x:v[x] for x in ("finite_training","finite_test","mae","baseline_mae")} for k,v in r["targets"].items()}, "prediction": predict_configuration(r,{"N_estimators":40,"criterion":"squared_error","Min_sample_leaves":3,"Min_impurity_decrease":2.75,"Max_depth":10})}, sort_keys=True))
