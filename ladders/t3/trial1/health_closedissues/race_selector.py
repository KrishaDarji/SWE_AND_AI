"""Independent Treatment 3 race: learned selector vs historical baseline."""
import csv, math, os, time
from pathlib import Path
from statistics import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATASET = str(Path(__file__).resolve().parents[8] / "moot/optimize/hpo/Health-ClosedIssues0000.csv")
FEATURES = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
TARGET = "ACC+"

def load_dataset(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(k not in rows[0] for k in FEATURES + [TARGET]):
        raise ValueError("missing required columns or empty dataset")
    data = []
    for i, r in enumerate(rows):
        try:
            x = {"N_estimators": float(r["N_estimators"]), "criterion": r["criterion"],
                 "Min_sample_leaves": float(r["Min_sample_leaves"]),
                 "Min_impurity_decrease": float(r["Min_impurity_decrease"]),
                 "Max_depth": float(r["Max_depth"])}
            y = float(r[TARGET])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("invalid row %d" % (i + 2)) from e
        if not math.isfinite(y) or not 0 <= y <= 1:
            raise ValueError("ACC+ outside [0,1] at row %d" % (i + 2))
        data.append((x, y))
    return data

def _pipeline():
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [1])], remainder="passthrough")
    return Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=80, max_depth=10, random_state=17, n_jobs=1))])

def evaluate(data, warmup=1000, block=100):
    if len(data) <= warmup + block: raise ValueError("not enough rows")
    learned_actual, baseline_actual, learned_pred, baseline_pred = [], [], [], []
    history = []
    for start in range(warmup, len(data), block):
        end = min(start + block, len(data))
        train = data[:start]
        rows = lambda items: [[x["N_estimators"], x["criterion"], x["Min_sample_leaves"], x["Min_impurity_decrease"], x["Max_depth"]] for x, _ in items]
        model = _pipeline().fit(rows(train), [y for _, y in train])
        candidates = rows(data[start:end])
        chosen = int(max(range(len(candidates)), key=lambda i: model.predict([candidates[i]])[0]))
        # Baseline: best mean observed ACC for the exact configuration; global mean fallback.
        stats = {}
        for x, y in train:
            key = tuple(sorted(x.items())); stats.setdefault(key, []).append(y)
        global_mean = mean(y for _, y in train)
        bchosen = int(max(range(len(candidates)), key=lambda i: mean(stats.get(tuple(sorted(data[start + i][0].items())), [global_mean]))) if candidates else 0)
        learned_actual.append(data[start + chosen][1]); baseline_actual.append(data[start + bchosen][1])
        learned_pred.append(float(model.predict([candidates[chosen]])[0])); baseline_pred.append(float(mean(stats.get(tuple(sorted(data[start + bchosen][0].items())), [global_mean]))))
        history.extend(data[start:end])
    return {
        "learned_selector": {"selected_mean_acc": mean(learned_actual), "mae": mean_absolute_error(learned_actual, learned_pred), "rmse": mean_squared_error(learned_actual, learned_pred) ** 0.5},
        "historical_baseline": {"selected_mean_acc": mean(baseline_actual), "mae": mean_absolute_error(baseline_actual, baseline_pred), "rmse": mean_squared_error(baseline_actual, baseline_pred) ** 0.5},
        "n_decisions": len(learned_actual), "warmup": warmup, "block": block,
    }

def main():
    t = time.perf_counter(); data = load_dataset(DATASET); metrics = evaluate(data); elapsed = time.perf_counter() - t
    print("rows=%d decisions=%d warmup=%d block=%d" % (len(data), metrics["n_decisions"], metrics["warmup"], metrics["block"]))
    for name, values in metrics.items():
        if isinstance(values, dict): print("%s selected_mean_acc=%.6f mae=%.6f rmse=%.6f" % (name, values["selected_mean_acc"], values["mae"], values["rmse"]))
    winner = "learned_selector" if metrics["learned_selector"]["selected_mean_acc"] > metrics["historical_baseline"]["selected_mean_acc"] else "historical_baseline"
    print("winner_by_selected_mean_acc=%s runtime_seconds=%.6f" % (winner, elapsed))

if __name__ == "__main__": main()
