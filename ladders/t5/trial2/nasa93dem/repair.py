#!/usr/bin/env python3
"""Training-only categorical effort prediction and minimum-change repair."""
import csv, math, random
from collections import defaultdict

DESCRIPTORS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced")
EFFORT = "Effort-"

def norm(v):
    return "" if v is None else str(v).strip()

def finite_number(v):
    try:
        x = float(norm(v))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = set(DESCRIPTORS) | {"idX", EFFORT}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("dataset is missing required columns")
        out = []
        for row_number, row in enumerate(reader, 2):
            descriptors = tuple(norm(row.get(k)) for k in DESCRIPTORS)
            effort = finite_number(row.get(EFFORT))
            if effort is not None and all(descriptors):
                out.append({"csv_row": row_number, "id": norm(row.get("idX")), "descriptors": descriptors, "effort": effort})
        if not out:
            raise ValueError("no valid rows")
        return out

def grouped_split(rows, seed=42, train_fraction=.75):
    groups = defaultdict(list)
    for row in rows:
        groups[row["descriptors"]].append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = len(rows) * train_fraction
    train, test = [], []
    total = 0
    for key in keys:
        bucket = groups[key]
        if total < target or not train:
            train.extend(bucket); total += len(bucket)
        else:
            test.extend(bucket)
    if not test and len(keys) > 1:
        key = keys[-1]
        moved = groups[key]
        train = [r for r in train if r not in moved]
        test.extend(moved)
    return train, test

class Estimator:
    def __init__(self, rows):
        if not rows:
            raise ValueError("insufficient training data")
        self.rows = list(rows)
        self.by_config = defaultdict(list)
        for row in rows:
            self.by_config[row["descriptors"]].append(row["effort"])
        self.means = {k: sum(v) / len(v) for k, v in self.by_config.items()}
        self.global_mean = sum(r["effort"] for r in rows) / len(rows)

    def predict(self, descriptors):
        d = tuple(norm(x) for x in descriptors)
        if len(d) != len(DESCRIPTORS) or not all(d):
            raise ValueError("incomplete or unseen proposal descriptors")
        if d in self.means:
            return self.means[d]
        scored = []
        for config, mean in self.means.items():
            distance = sum(a != b for a, b in zip(d, config))
            scored.append((distance, mean))
        if not scored:
            raise ValueError("no training configurations")
        best = min(x[0] for x in scored)
        values = [x[1] for x in scored if x[0] == best]
        return sum(values) / len(values)

def repair(proposal, budget, train_rows):
    if finite_number(budget) is None:
        raise ValueError("budget must be finite numeric")
    budget = float(budget)
    if len(proposal) != len(DESCRIPTORS):
        raise ValueError("proposal must contain all 22 descriptors")
    original = tuple(norm(x) for x in proposal)
    if not all(original):
        raise ValueError("proposal contains missing descriptor")
    model = Estimator(train_rows)
    original_effort = model.predict(original)
    base = {"original_descriptors": original, "original_estimated_effort": original_effort, "budget": budget}
    if original_effort <= budget:
        return dict(base, status="no-repair-needed", predicted_feasible=True)
    candidates = []
    for row in train_rows:
        candidate = row["descriptors"]
        predicted = model.predict(candidate)
        if predicted <= budget:
            changes = sum(a != b for a, b in zip(original, candidate))
            candidates.append((changes, predicted, row["csv_row"], row, candidate))
    if not candidates:
        return dict(base, status="no-feasible-repair", predicted_feasible=False)
    _, predicted, _, row, candidate = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    changed = [{"field": name, "old": old, "new": new} for name, old, new in zip(DESCRIPTORS, original, candidate) if old != new]
    return dict(base, status="repair-found", predicted_feasible=True, candidate_descriptors=candidate, candidate_estimated_effort=predicted, candidate_id=row["id"], candidate_csv_row=row["csv_row"], changed_fields=changed, change_count=len(changed), change_count_is_proxy=True)

def run(path, budget=100):
    rows = load_rows(path)
    train, test = grouped_split(rows)
    if not test:
        raise ValueError("split produced no held-out rows")
    result = repair(test[0]["descriptors"], budget, train)
    result.update(train_rows=len(train), test_rows=len(test), proposal_id=test[0]["id"], proposal_csv_row=test[0]["csv_row"], train_ids=[r["id"] for r in train])
    return result

def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--budget", type=float, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.budget), sort_keys=True))

if __name__ == "__main__":
    main()
