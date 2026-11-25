"""Training-only categorical effort estimation and minimum-change repair."""
import csv, math, random
from collections import defaultdict

TARGET = "Effort-"
EXCLUDED = {"Kloc+", "Effort-", "Defects-"}

def clean(value):
    return value.strip() if isinstance(value, str) else value

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        raw = csv.DictReader(fh)
        fields = raw.fieldnames or []
        desc = [f for f in fields if not f.endswith("X") and f not in EXCLUDED]
        rows, invalid = [], []
        for number, row in enumerate(raw, 2):
            try:
                effort = float(clean(row[TARGET]))
                if not math.isfinite(effort): raise ValueError
            except (KeyError, TypeError, ValueError):
                invalid.append((number, row.get("idX")))
                continue
            rows.append({"csv_line": number, "idX": clean(row.get("idX")),
                         "descriptors": {k: clean(row.get(k, "")) for k in desc},
                         "effort": effort})
    if not desc: raise ValueError("no usable descriptor columns")
    return desc, rows, invalid

def grouped_split(rows, seed=42, train_fraction=.75):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row["descriptors"].items())].append(row)
    keys = list(groups)
    if len(keys) < 2: raise ValueError("insufficient data: need at least two distinct descriptor groups")
    random.Random(seed).shuffle(keys)
    target = len(rows) * train_fraction
    train, test = [], []
    for index, key in enumerate(keys):
        bucket = groups[key]
        # Keep at least one complete group for test; never move individual rows.
        if index < len(keys) - 1 and (not train or len(train) + len(bucket) <= target):
            train.extend(bucket)
        else:
            test.extend(bucket)
    if not train or not test: raise ValueError("insufficient data: split must have nonempty train and test sets")
    return train, test

class Estimator:
    def __init__(self, descriptors, rows):
        if len(rows) < 1: raise ValueError("insufficient training data")
        self.descriptors = list(descriptors)
        self.rows = list(rows)
        self.by_config = defaultdict(list)
        for row in rows: self.by_config[self.key(row["descriptors"])].append(row["effort"])
        self.global_value = sum(r["effort"] for r in rows) / len(rows)

    def key(self, proposal):
        missing = [k for k in self.descriptors if k not in proposal]
        if missing: raise ValueError("incomplete proposal: " + ",".join(missing))
        return tuple(clean(proposal[k]) for k in self.descriptors)

    def predict(self, proposal):
        key = self.key(proposal)
        values = self.by_config.get(key)
        if values: return sum(values) / len(values)
        # A transparent categorical nearest-neighbour fallback for unseen proposals.
        scored = sorted((sum(proposal[k] != r["descriptors"][k] for k in self.descriptors), r["effort"])
                        for r in self.rows)
        best = scored[0][0]
        vals = [v for d, v in scored if d == best]
        return sum(vals) / len(vals)

def repair(proposal, budget, estimator):
    if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite nonnegative number")
    original = {k: clean(v) for k, v in proposal.items()}
    original_effort = estimator.predict(original)
    if original_effort <= budget:
        return {"status": "no-repair-needed", "original": original, "candidate": original,
                "changed": [], "change_count": 0, "original_effort": original_effort,
                "candidate_effort": original_effort, "budget": budget, "feasible": True}
    candidates = []
    for row in estimator.rows:
        candidate = dict(row["descriptors"])
        predicted = estimator.predict(candidate)
        if predicted <= budget:
            changed = [(k, original[k], candidate[k]) for k in estimator.descriptors if original[k] != candidate[k]]
            candidates.append((len(changed), predicted, row["csv_line"], candidate, changed))
    if not candidates:
        return {"status": "no-feasible-repair", "original": original, "candidate": None,
                "changed": [], "change_count": None, "original_effort": original_effort,
                "candidate_effort": None, "budget": budget, "feasible": False}
    count, predicted, line, candidate, changed = min(candidates, key=lambda x: x[:3])
    return {"status": "repaired", "original": original, "candidate": candidate, "changed": changed,
            "change_count": count, "original_effort": original_effort, "candidate_effort": predicted,
            "budget": budget, "feasible": True, "candidate_csv_line": line}

def validate_proposal(proposal, descriptors):
    if not isinstance(proposal, dict): raise ValueError("proposal must be a mapping")
    missing = [k for k in descriptors if k not in proposal or clean(proposal[k]) == ""]
    if missing: raise ValueError("missing/unseen descriptors: " + ",".join(missing))
