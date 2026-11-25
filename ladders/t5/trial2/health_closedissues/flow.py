"""Streaming validation summaries for configuration records."""
import csv
import copy
import math

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth")
_DIRECTION = {"MRE-": "min", "ACC+": "max", "PRED40+": "max"}


def _number(text):
    if text is None or str(text).strip() == "":
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _config(row):
    return {"N_estimators": int(row["N_estimators"]),
            "criterion": row["criterion"],
            "Min_sample_leaves": int(row["Min_sample_leaves"]),
            "Min_impurity_decrease": float(row["Min_impurity_decrease"]),
            "Max_depth": int(row["Max_depth"])}


class FlowEvaluator:
    """Consumes rows once; each snapshot describes only rows seen so far."""
    def __init__(self):
        self._states = {name: {"count": 0, "mean": None, "m2": 0.0,
                               "best": None} for name in OUTCOMES}
        self._snapshots = []
        self._next_id = 1

    def accept(self, row, record_id=None):
        rid = self._next_id if record_id is None else record_id
        self._next_id = max(self._next_id, rid + 1)
        for name in OUTCOMES:
            value = _number(row.get(name, ""))
            if value is None:
                continue
            state = self._states[name]
            state["count"] += 1
            delta = value - (state["mean"] or 0.0)
            state["mean"] = (value if state["count"] == 1 else
                              state["mean"] + delta / state["count"])
            state["m2"] += delta * (value - state["mean"])
            best = state["best"]
            better = best is None or ((_DIRECTION[name] == "min" and value < best["value"]) or
                                      (_DIRECTION[name] == "max" and value > best["value"]))
            if better:
                state["best"] = {"value": value, "record_id": rid,
                                  "configuration": copy.deepcopy(_config(row))}
        self._snapshots.append(self.summary())

    def summary(self):
        result = {}
        for name, state in self._states.items():
            result[name] = {"count": state["count"], "mean": state["mean"],
                            "sample_variance": (state["m2"] / (state["count"] - 1)
                                                if state["count"] > 1 else None),
                            "best": copy.deepcopy(state["best"])}
        return copy.deepcopy(result)

    @property
    def snapshots(self):
        return copy.deepcopy(self._snapshots)


def evaluate_csv(path):
    evaluator = FlowEvaluator()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            evaluator.accept(row)
    return evaluator


def write_metrics(path, evaluator):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fields = ["step", "record_id"] + [x for name in OUTCOMES for x in
                 (name + "_count", name + "_mean", name + "_sample_variance")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, snapshot in enumerate(evaluator.snapshots, 1):
            row = {"step": index, "record_id": index}
            for name in OUTCOMES:
                state = snapshot[name]
                row.update({name + "_count": state["count"],
                            name + "_mean": state["mean"],
                            name + "_sample_variance": state["sample_variance"]})
            writer.writerow(row)
