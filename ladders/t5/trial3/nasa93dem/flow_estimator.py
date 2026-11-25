#!/usr/bin/env python3
"""Sequential, leakage-resistant project-effort assessment."""
import csv
import math
from collections import defaultdict

DESCRIPTORS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
               "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
               "apex", "plex", "ltex", "tool", "site", "sced")
OUTCOME = "Effort-"

def clean(value):
    value = "" if value is None else str(value).strip()
    return "?" if value == "" or value == "?" else value

def finite_effort(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def record_id(row, ordinal):
    """IDs are the trimmed idX field, falling back to one-based CSV position."""
    value = clean(row.get("idX"))
    return value if value != "?" else str(ordinal)

class CategoricalRegressor:
    """Nearest categorical rows; ties predict their revealed-effort mean."""
    def __init__(self, descriptors=DESCRIPTORS):
        self.descriptors = tuple(descriptors)
        self.rows = []
        self.categories = {name: set() for name in self.descriptors}

    def fit(self, rows):
        self.rows = []
        self.categories = {name: set() for name in self.descriptors}
        for features, effort in rows:
            normalized = tuple(clean(features.get(name)) for name in self.descriptors)
            self.rows.append((normalized, effort))
            for name, value in zip(self.descriptors, normalized):
                if value != "?":
                    self.categories[name].add(value)
        return self

    def _distance(self, left, right):
        gaps = []
        for name, a, b in zip(self.descriptors, left, right):
            # EZR's symbolic distance: equal=0, unequal=1, two unknowns=1.
            if a == "?" and b == "?":
                gap = 1.0
            elif a == "?" or b == "?":
                gap = 1.0
            else:
                gap = 0.0 if a == b else 1.0
            gaps.append(gap)
        return (sum(gap ** 2 for gap in gaps) / len(gaps)) ** 0.5

    def predict(self, features):
        if not self.rows:
            raise ValueError("cannot predict without revealed training rows")
        query = tuple(clean(features.get(name)) for name in self.descriptors)
        distances = [(self._distance(query, values), effort) for values, effort in self.rows]
        nearest = min(item[0] for item in distances)
        tied = [effort for distance, effort in distances if distance == nearest]
        return sum(tied) / len(tied)

def assess(rows):
    """Reveal rows in order and return immutable per-arrival assessment records."""
    history, output, valid_seen = [], [], 0
    for ordinal, row in enumerate(rows, 1):
        effort = finite_effort(row.get(OUTCOME))
        features = {name: clean(row.get(name)) for name in DESCRIPTORS}
        predicted = baseline = None
        if valid_seen >= 2 and history:
            model = CategoricalRegressor().fit(history)
            predicted = model.predict(features)
            baseline = sum(item[1] for item in history) / len(history)
        if effort is not None:
            if predicted is not None:
                output.append({"arrival_id": record_id(row, ordinal),
                               "prior_training_count": len(history),
                               "prediction": predicted, "baseline": baseline,
                               "revealed_effort": effort,
                               "absolute_error": abs(predicted-effort),
                               "baseline_absolute_error": abs(baseline-effort)})
            history.append((features, effort))
            valid_seen += 1
    cumulative = 0.0
    base_cumulative = 0.0
    for item in output:
        cumulative += item["absolute_error"]
        base_cumulative += item["baseline_absolute_error"]
        item["cumulative_mae"] = cumulative / output.index(item).__add__(1)
        item["baseline_cumulative_mae"] = base_cumulative / output.index(item).__add__(1)
    return output

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def write_metrics(records, path):
    fields = ("arrival_id", "prior_training_count", "prediction", "baseline",
              "revealed_effort", "absolute_error", "baseline_absolute_error",
              "cumulative_mae", "baseline_cumulative_mae")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
