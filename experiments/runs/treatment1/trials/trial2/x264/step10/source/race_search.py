"""Independent race-vs-baseline search for X264 configuration data."""
import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    config: tuple
    score: float


def load_records(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or fields[-1] != "Performance-":
            raise ValueError("expected Performance- as the final CSV column")
        records = []
        for row in reader:
            config = tuple(int(row[name]) for name in fields[:-1])
            records.append(Record(config, float(row[fields[-1]])))
    if not records:
        raise ValueError("dataset is empty")
    return records


def _validate_budget(records, budget):
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > len(records):
        raise ValueError("budget cannot exceed dataset size")


def baseline_search(records, budget, seed=7):
    """Random-search baseline: inspect exactly budget unique records."""
    _validate_budget(records, budget)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    chosen = [records[index] for index in order[:budget]]
    return min(chosen, key=lambda record: record.score), budget


def race_search(records, budget, seed=7, batch_size=16, survivor_count=4):
    """Successive-halving race with a fixed total evaluation budget.

    Each round evaluates an equal number of new candidates.  The best
    survivors seed the next round; unevaluated records fill each round.
    """
    _validate_budget(records, budget)
    if batch_size <= 0 or survivor_count <= 0 or survivor_count >= batch_size:
        raise ValueError("require 0 < survivor_count < batch_size")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    cursor = 0
    evaluated = []
    survivors = []
    while len(evaluated) < budget:
        remaining = budget - len(evaluated)
        take = min(batch_size, remaining)
        indices = order[cursor:cursor + take]
        cursor += take
        round_records = [records[index] for index in indices]
        evaluated.extend(round_records)
        ranked = sorted(survivors + round_records, key=lambda record: record.score)
        survivors = ranked[:min(survivor_count, len(ranked))]
    return min(evaluated, key=lambda record: record.score), len(evaluated)


def compare(path, budget=128, seed=7):
    records = load_records(path)
    baseline, baseline_evaluations = baseline_search(records, budget, seed)
    raced, race_evaluations = race_search(records, budget, seed)
    return {
        "dataset_rows": len(records),
        "budget": budget,
        "baseline": baseline,
        "baseline_evaluations": baseline_evaluations,
        "race": raced,
        "race_evaluations": race_evaluations,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()
    result = compare(args.dataset, args.budget)
    print("dataset_rows={}".format(result["dataset_rows"]))
    print("evaluation_budget={}".format(result["budget"]))
    print("baseline_evaluations={} best={:.3f}".format(result["baseline_evaluations"], result["baseline"].score))
    print("race_evaluations={} best={:.3f}".format(result["race_evaluations"], result["race"].score))
    print("race_minus_baseline={:.3f}".format(result["race"].score - result["baseline"].score))
