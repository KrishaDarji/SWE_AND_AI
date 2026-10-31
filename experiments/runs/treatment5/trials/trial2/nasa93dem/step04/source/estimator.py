"""Sequential project-effort assessment using only prior revealed projects."""
import csv, math

OUTCOME = "Effort-"
DESCRIPTORS = ("prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced")

def clean(v):
    return "" if v is None else str(v).strip()

def finite(v):
    try:
        x = float(clean(v))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _distance(a, b, categories):
    d = 0.0
    n = 0
    for k in DESCRIPTORS:
        av, bv = clean(a.get(k)), clean(b.get(k))
        if not av or not bv or av == "?" or bv == "?":
            continue
        n += 1
        if av != bv:
            d += 1.0
    return d / n if n else 1.0

def _predict(history, row):
    """A deterministic 1-nearest-neighbor estimate; unseen categories remain unequal."""
    if not history:
        return None
    best = min(history, key=lambda h: _distance(h["row"], row, None))
    return best["effort"]

def assess(rows):
    """Return frozen records. First two valid rows are warm-up and unscored."""
    history, results = [], []
    cumulative_model = cumulative_base = 0.0
    scored = 0
    for pos, row in enumerate(rows, 1):
        effort = finite(row.get(OUTCOME))
        prior = len(history)
        prediction = _predict(history, row)
        baseline = (sum(h["effort"] for h in history) / prior) if prior else None
        record = {"arrival": clean(row.get("idX")) or str(pos), "prior_training_count": prior,
                  "prediction": prediction, "baseline": baseline, "revealed_effort": effort,
                  "prediction_abs_error": None, "baseline_abs_error": None,
                  "cumulative_prediction_mae": None, "cumulative_baseline_mae": None,
                  "warmup": prior < 2, "valid_outcome": effort is not None}
        if effort is not None and prior >= 2 and prediction is not None and baseline is not None:
            pe, be = abs(prediction-effort), abs(baseline-effort)
            cumulative_model += pe; cumulative_base += be; scored += 1
            record.update(prediction_abs_error=pe, baseline_abs_error=be,
                          cumulative_prediction_mae=cumulative_model/scored,
                          cumulative_baseline_mae=cumulative_base/scored)
        results.append(record)
        if effort is not None:
            history.append({"row": {k: clean(row.get(k)) for k in DESCRIPTORS}, "effort": effort})
    return results

def write_metrics(rows, path):
    fields = ["arrival","prior_training_count","prediction","baseline","revealed_effort","prediction_abs_error","baseline_abs_error","cumulative_prediction_mae","cumulative_baseline_mae","warmup","valid_outcome"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sequential NASA93 effort assessment")
    p.add_argument("dataset"); p.add_argument("--metrics", required=True)
    a = p.parse_args(); write_metrics(assess(read_rows(a.dataset)), a.metrics)
