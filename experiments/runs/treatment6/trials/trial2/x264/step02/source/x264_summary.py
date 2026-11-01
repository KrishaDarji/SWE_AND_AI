"""Summarize historical X264 encoder settings and measured outcomes."""
import csv, math, statistics, sys, random

MISSING = {"", "?", "nil", "NA", "N/A", "null"}

def _number(value):
    text = value.strip()
    if text in MISSING:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None

def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return [], [], []
    header = [cell.strip() for cell in rows[0]]
    records = []
    invalid = []
    for line, row in enumerate(rows[1:], 2):
        if not any(cell.strip() for cell in row):
            continue
        values = row + [""] * max(0, len(header) - len(row))
        values = values[:len(header)]
        records.append(dict(zip(header, values)))
    return header, records, invalid

def summarize(path):
    header, records, _ = load(path)
    outcome_names = [name for name in header if name.endswith("-")]
    outcome = outcome_names[0] if outcome_names else None
    settings = [name for name in header if name != outcome]
    frequencies = {}
    for name in settings:
        counts = {}
        for record in records:
            value = record.get(name, "").strip()
            key = value if value not in MISSING else "<missing>"
            counts[key] = counts.get(key, 0) + 1
        frequencies[name] = counts
    outcomes = []
    missing = 0
    invalid = 0
    if outcome:
        for record in records:
            raw = record.get(outcome, "")
            value = _number(raw)
            if raw.strip() in MISSING:
                missing += 1
            elif value is None:
                invalid += 1
            else:
                outcomes.append(value)
    stats = None
    if outcomes:
        stats = {"count": len(outcomes), "min": min(outcomes),
                 "max": max(outcomes), "mean": statistics.fmean(outcomes),
                 "median": statistics.median(outcomes),
                 "stdev": statistics.stdev(outcomes) if len(outcomes) > 1 else 0.0}
    return {"record_count": len(records), "headers": header, "settings": settings,
            "outcome": outcome, "direction": "lower is better" if outcome else None,
            "frequencies": frequencies, "outcome_stats": stats,
            "missing_outcomes": missing, "invalid_outcomes": invalid,
            "missing_input": not bool(header)}

def render(summary):
    lines = ["Historical X264 run summary", "record_count: %d" % summary["record_count"],
             "outcome: %s (%s)" % (summary["outcome"] or "none", summary["direction"] or "unavailable")]
    if summary["outcome_stats"]:
        lines.append("outcome_stats: " + ", ".join("%s=%.6g" % item for item in summary["outcome_stats"].items()))
    else:
        lines.append("outcome_stats: unavailable")
    lines.append("missing_outcomes: %d; invalid_outcomes: %d" % (summary["missing_outcomes"], summary["invalid_outcomes"]))
    lines.append("setting_frequencies:")
    for name, counts in summary["frequencies"].items():
        lines.append("  %s: %s" % (name, ", ".join("%s=%d" % pair for pair in sorted(counts.items()))))
    return "\n".join(lines)

def _groups(records, settings):
    groups = {}
    for record in records:
        key = tuple(record.get(name, "").strip() for name in settings)
        groups.setdefault(key, []).append(record)
    return [(key, groups[key]) for key in sorted(groups)]

def _target(record, outcome):
    return _number(record.get(outcome, ""))

def split_groups(records, settings, seed=42):
    groups = _groups(records, settings)
    if len(groups) < 2: raise ValueError("at least two configuration groups are required")
    order = list(range(len(groups))); random.Random(seed).shuffle(order)
    cut = min(max(1, int(round(len(groups) * .8))), len(groups) - 1)
    train_keys = {groups[i][0] for i in order[:cut]}
    train = [r for key, rows in groups if key in train_keys for r in rows]
    test = [r for key, rows in groups if key not in train_keys for r in rows]
    return train, test

class NearestNeighborModel:
    """1-NN on training-fitted numeric/categorical normalized settings."""
    def __init__(self, settings, outcome): self.settings, self.outcome = settings, outcome
    def fit(self, records):
        valid = [(r, _target(r, self.outcome)) for r in records if _target(r, self.outcome) is not None]
        if not valid: raise ValueError("no finite training targets")
        self.rows = [r for r, _ in valid]; self.values = [y for _, y in valid]
        self.encoders = {}
        for name in self.settings:
            vals = sorted({r.get(name, "").strip() for r in self.rows})
            self.encoders[name] = {v: i / max(1, len(vals)-1) for i, v in enumerate(vals)}
        self.mean = statistics.fmean(self.values); return self
    def _vector(self, record):
        missing = [n for n in self.settings if n not in record]
        extra = [n for n in record if n not in self.settings and n != self.outcome]
        if missing or extra: raise ValueError("configuration fields must exactly match settings")
        out = []
        for name in self.settings:
            value = record.get(name, "").strip()
            if value in MISSING: raise ValueError("missing setting: %s" % name)
            if value not in self.encoders[name]: raise ValueError("unseen setting value for %s" % name)
            out.append(self.encoders[name][value])
        return out
    def predict_one(self, record):
        try: vector = self._vector(record)
        except ValueError as exc:
            if str(exc).startswith("unseen setting value"): return self.mean
            raise
        best = min(range(len(self.rows)), key=lambda i: sum((a-b)**2 for a,b in zip(vector, self._vector(self.rows[i]))))
        return self.values[best]
    def predict(self, records): return [self.predict_one(r) for r in records]

def evaluate(path, seed=42):
    header, records, _ = load(path); outcome = "Performance-"; settings = [n for n in header if n != outcome]
    train, test = split_groups(records, settings, seed); model = NearestNeighborModel(settings, outcome).fit(train)
    valid_test = [(r, _target(r, outcome)) for r in test if _target(r, outcome) is not None]
    predictions = [model.predict_one(r) for r, _ in valid_test]
    mae = statistics.fmean(abs(p-y) for p, (_, y) in zip(predictions, valid_test)) if valid_test else None
    baseline = statistics.fmean(model.values)
    base_mae = statistics.fmean(abs(b-y) for _, (_, y) in zip(valid_test, valid_test)) if False else (statistics.fmean(abs(baseline-y) for _, y in valid_test) if valid_test else None)
    return {"model": model, "settings": settings, "train_rows": len(train), "test_rows": len(test), "train_groups": len(_groups(train, settings)), "test_groups": len(_groups(test, settings)), "evaluated_test_rows": len(valid_test), "mae": mae, "baseline_mae": base_mae}

def render_prediction(result, configuration):
    return "prediction: %.6f\ntrain_rows: %d; test_rows: %d; train_groups: %d; test_groups: %d\nheld_out_mae: %s; baseline_mae: %s" % (result["model"].predict_one(configuration), result["train_rows"], result["test_rows"], result["train_groups"], result["test_groups"], result["mae"], result["baseline_mae"])

def main(argv):
    if len(argv) != 2:
        print("usage: x264_summary.py DATASET", file=sys.stderr); return 2
    try:
        if len(argv) == 2: print(render(summarize(argv[1])))
        else:
            result = evaluate(argv[1]); config = {n: "0" for n in result["settings"]}
            print(render_prediction(result, config))
        return 0
    except (OSError, csv.Error) as exc:
        print("error: %s" % exc, file=sys.stderr); return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
