#!/usr/bin/env python3
"""Leakage-safe sequential categorical effort assessment."""
import csv, math, sys

DESCRIPTORS = ["prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced"]
OUTCOME = "Effort-"

def clean(value):
    value = "" if value is None else str(value).strip()
    return None if value in ("", "?") else value.lower()

def finite_effort(value):
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = DESCRIPTORS + [OUTCOME]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("missing required columns: " + ", ".join(missing))
        for number, row in enumerate(reader, 1):
            yield "arrival-%04d" % number, row

def distance(left, right):
    total = 0.0
    for name in DESCRIPTORS:
        a, b = clean(left.get(name)), clean(right.get(name))
        total += 0.0 if a is not None and b is not None and a == b else 1.0
    return total / len(DESCRIPTORS)

def predict(history, row):
    if not history:
        return None
    nearest = min(distance(row, old) for old, _ in history)
    values = [effort for old, effort in history if distance(row, old) == nearest]
    return sum(values) / len(values)

def assess(rows):
    history, results = [], []
    score_count = 0
    prediction_error_sum = baseline_error_sum = 0.0
    for identifier, row in rows:
        # Predictor reads and frozen estimates precede outcome access.
        prediction = predict(history, row) if len(history) >= 2 else None
        baseline = sum(effort for _, effort in history) / len(history) if len(history) >= 2 else None
        actual = finite_effort(row.get(OUTCOME))
        result = {"arrival_id": identifier, "prior_training_count": len(history)}
        if actual is None:
            result.update(status="invalid-effort", revealed_effort="")
        elif len(history) < 2:
            result.update(status="warm-up", revealed_effort=actual)
            history.append((dict(row), actual))
        else:
            prediction_error = abs(prediction - actual)
            baseline_error = abs(baseline - actual)
            prediction_error_sum += prediction_error
            baseline_error_sum += baseline_error
            score_count += 1
            result.update(status="scored", prediction=prediction, baseline=baseline,
                          revealed_effort=actual, absolute_error=prediction_error,
                          baseline_absolute_error=baseline_error,
                          cumulative_mae=prediction_error_sum / score_count,
                          baseline_cumulative_mae=baseline_error_sum / score_count)
            history.append((dict(row), actual))
        results.append(result)
    return results

def write_csv(results, path):
    fields = ["arrival_id", "prior_training_count", "status", "prediction", "baseline", "revealed_effort", "absolute_error", "baseline_absolute_error", "cumulative_mae", "baseline_cumulative_mae"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)

def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: arrival_estimator.py DATASET OUTPUT_CSV")
    write_csv(assess(read_rows(argv[1])), argv[2])
    print("processed=streamed")

if __name__ == "__main__":
    main(sys.argv)
