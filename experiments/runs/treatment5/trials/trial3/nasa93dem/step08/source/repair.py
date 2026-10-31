#!/usr/bin/env python3
"""Training-only, categorical descriptor repair for NASA93-style proposals."""
import csv, math, random
from pathlib import Path

DESCRIPTORS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced")
TARGET = "Effort-"

class InputError(ValueError):
    pass

def norm(value):
    if value is None:
        return ""
    return " ".join(str(value).split())

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(x not in reader.fieldnames for x in DESCRIPTORS + (TARGET,)):
            raise InputError("missing required CSV columns")
        rows = []
        for number, raw in enumerate(reader, 2):
            try:
                target = float(norm(raw[TARGET]))
                if not math.isfinite(target):
                    raise ValueError
            except (TypeError, ValueError):
                continue
            row = {key: norm(raw.get(key, "")) for key in DESCRIPTORS}
            row["_target"] = target
            row["_csv_row"] = number
            row["_id"] = norm(raw.get("idX", ""))
            if any(not row[key] for key in DESCRIPTORS):
                continue
            rows.append(row)
    return rows

def split_grouped(rows, seed=42, train_fraction=.75):
    if len(rows) < 2:
        raise InputError("insufficient valid rows")
    groups = {}
    for row in rows:
        groups.setdefault(config_key(row), []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    cut = max(1, min(len(keys) - 1, int(len(keys) * train_fraction)))
    train_keys = set(keys[:cut])
    train = [r for r in rows if config_key(r) in train_keys]
    test = [r for r in rows if config_key(r) not in train_keys]
    if not train or not test:
        raise InputError("grouped split produced an empty partition")
    return train, test

def config_key(row):
    return tuple(norm(row.get(key, "")) for key in DESCRIPTORS)

def fit(train):
    if not train:
        raise InputError("insufficient training data")
    by_config = {}
    for row in train:
        by_config.setdefault(config_key(row), []).append(row["_target"])
    means = {key: sum(values) / len(values) for key, values in by_config.items()}
    overall = sum(r["_target"] for r in train) / len(train)
    return {"means": means, "overall": overall, "train": train}

def estimate(model, row):
    return model["means"].get(config_key(row), model["overall"])

def repair(model, proposal, budget):
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        raise InputError("budget must be finite and nonnegative")
    if not math.isfinite(budget) or budget < 0:
        raise InputError("budget must be finite and nonnegative")
    if any(not norm(proposal.get(key, "")) for key in DESCRIPTORS):
        raise InputError("proposal is incomplete")
    original = {key: norm(proposal[key]) for key in DESCRIPTORS}
    original_effort = estimate(model, original)
    if original_effort <= budget:
        return {"status": "no-repair-needed", "original": original, "candidate": original, "changed": [], "original_effort": original_effort, "estimated_effort": original_effort, "budget": budget, "feasible": True}
    candidates = []
    for row in model["train"]:
        key = config_key(row)
        predicted = model["means"][key]
        if predicted <= budget:
            changed = [name for name in DESCRIPTORS if norm(row[name]) != original[name]]
            candidates.append((len(changed), predicted, row["_csv_row"], row, changed))
    if not candidates:
        return {"status": "no-feasible-repair", "original": original, "candidate": None, "changed": [], "original_effort": original_effort, "estimated_effort": None, "budget": budget, "feasible": False}
    _, predicted, _, row, changed = min(candidates, key=lambda item: item[:3])
    candidate = {key: norm(row[key]) for key in DESCRIPTORS}
    return {"status": "repair", "original": original, "candidate": candidate, "changed": [{"field": key, "old": original[key], "new": candidate[key]} for key in changed], "original_effort": original_effort, "estimated_effort": predicted, "budget": budget, "feasible": True, "candidate_csv_row": row["_csv_row"], "candidate_id": row.get("_id", "")}

def run(path):
    rows = read_rows(path)
    train, test = split_grouped(rows)
    model = fit(train)
    proposal = test[0]
    result = repair(model, proposal, 100)
    result.update({"train_rows": len(train), "test_rows": len(test), "proposal_csv_row": proposal["_csv_row"], "proposal_id": proposal.get("_id", "")})
    return result

def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.csv_path)), sort_keys=True))

if __name__ == "__main__":
    main()
