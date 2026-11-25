"""Independent categorical effort estimator and evidence report."""
import csv, math, random, statistics

DESCRIPTORS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced")
TARGET = "Effort-"

def clean(v):
    return " ".join(str(v).strip().split()).lower()

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for position, r in enumerate(raw):
        try:
            y = float(r[TARGET].strip())
            if not math.isfinite(y):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"id": r.get("idX", ""), "position": position,
                     "x": tuple(clean(r.get(c, "")) for c in DESCRIPTORS), "y": y})
    return rows

def grouped_split(rows, seed=42, train_fraction=.75):
    groups = {}
    for row in rows:
        groups.setdefault(row["x"], []).append(row)
    if len(groups) < 2:
        raise ValueError("insufficient descriptor groups for a split")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * train_fraction)
    cut = min(max(cut, 1), len(keys) - 1)
    train_keys = set(keys[:cut])
    train = [r for r in rows if r["x"] in train_keys]
    test = [r for r in rows if r["x"] not in train_keys]
    return train, test

def median(values):
    return statistics.median(values)

class CategoricalNearestMedian:
    """Hamming-distance k-neighbor median; ties resolve by original id/position."""
    def __init__(self, k=5):
        self.k = k
        self.rows = []
        self.reference = None

    def fit(self, rows):
        if not rows:
            raise ValueError("training data is empty")
        self.rows = list(rows)
        self.reference = median([r["y"] for r in self.rows])
        return self

    def neighbors(self, x):
        if len(x) != len(DESCRIPTORS):
            raise ValueError("proposal must contain exactly 22 descriptors")
        scored = []
        for r in self.rows:
            distance = sum(a != b for a, b in zip(x, r["x"]))
            scored.append((distance, str(r["id"]), r["position"], r))
        scored.sort(key=lambda z: (z[0], z[1], z[2]))
        return scored[:min(self.k, len(scored))]

    def predict(self, x):
        ns = self.neighbors(tuple(clean(v) for v in x))
        return median([z[3]["y"] for z in ns])

    def explain(self, x):
        ns = self.neighbors(tuple(clean(v) for v in x))
        estimate = median([z[3]["y"] for z in ns])
        return {"estimate": estimate, "training_reference_median": self.reference,
                "evidence": [{"label": "model neighbor", "id": z[3]["id"],
                              "distance": z[0], "effort": z[3]["y"]} for z in ns],
                "note": "Distances are mismatched descriptor counts; evidence is not causal contribution."}

def report(dataset, seed=42):
    rows = load_rows(dataset)
    if len({r["x"] for r in rows}) < 2:
        raise ValueError("insufficient descriptor groups for a split")
    train, test = grouped_split(rows, seed)
    if not test:
        raise ValueError("held-out set is empty")
    model = CategoricalNearestMedian().fit(train)
    target = min(test, key=lambda r: r["position"])
    explanation = model.explain(target["x"])
    explanation.update({"record_id": target["id"], "record_position": target["position"],
                        "descriptor_count": len(target["x"]), "train_rows": len(train),
                        "test_rows": len(test), "audit": {"observed_effort": target["y"]}})
    return {"train_rows": len(train), "test_rows": len(test), "groups": len({r["x"] for r in rows}),
            "selected": target, "pre_audit": explanation}

def render(result):
    e = result["pre_audit"]
    lines = ["EFFORT ESTIMATE (frozen before audit)", f"Held-out record id: {e['record_id']}",
             f"Descriptor inputs: {e['descriptor_count']} categorical ratings", f"Training rows: {e['train_rows']}; held-out rows: {e['test_rows']}",
             f"Model estimate: {e['estimate']}", f"Training-history reference (median): {e['training_reference_median']}",
             "Evidence labels: model neighbor (training-only)"]
    for n in e["evidence"]:
        lines.append(f"  id={n['id']} distance={n['distance']} effort={n['effort']}")
    lines += [e["note"], "AUDIT (revealed after estimate and explanation)", f"Measured held-out effort: {e['audit']['observed_effort']}",
              "Limitations: missing/unseen ratings are normalized to empty and remain mismatches; correlated factors can duplicate signal; this is a scope-limited estimate, not a causal claim, interval, cost, or guarantee."]
    return "\n".join(lines) + "\n"

def main(dataset):
    print(render(report(dataset)), end="")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python estimator.py DATASET")
    main(sys.argv[1])
