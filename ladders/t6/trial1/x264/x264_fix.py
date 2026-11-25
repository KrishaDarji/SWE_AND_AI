"""Find the smallest observed X264 setting change that improves a baseline."""
import csv
import json
import math
import sys

TARGET = "Performance-"

def _number(value):
    try:
        value = float(str(value).strip())
        return value if math.isfinite(value) else None
    except (AttributeError, TypeError, ValueError):
        return None

def _same(a, b):
    """Compare numeric settings numerically, while retaining text semantics."""
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a).strip() == str(b).strip()

def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if TARGET not in fields:
            raise ValueError("dataset must contain Performance-")
        settings = [name for name in fields if name != TARGET]
        rows = []
        for record_id, row in enumerate(reader):
            outcome = _number(row.get(TARGET, ""))
            if outcome is not None:
                rows.append((record_id, row, outcome))
        return settings, rows

def fix(path, baseline_id=0):
    settings, rows = _read(path)
    if not isinstance(baseline_id, int) or isinstance(baseline_id, bool) or baseline_id < 0:
        raise ValueError("baseline_id must be a nonnegative integer")
    all_ids = {record_id for record_id, _, _ in rows}
    if baseline_id not in all_ids:
        raise ValueError("baseline identifier has no finite Performance- outcome")
    baseline_id, baseline, baseline_outcome = next(x for x in rows if x[0] == baseline_id)
    candidates = []
    for record_id, row, outcome in rows:
        if outcome >= baseline_outcome:
            continue
        changed = [{"setting": name, "old": baseline.get(name, ""),
                    "new": row.get(name, "")} for name in settings
                   if not _same(baseline.get(name, ""), row.get(name, ""))]
        candidates.append((len(changed), outcome, record_id, row, changed))
    result = {"status": "no_repair" if not candidates else "repaired",
              "target": TARGET, "direction": "lower Performance- is better",
              "record_identifier": "zero-based CSV data-record position (header excluded)",
              "baseline": {"record_id": baseline_id, "configuration": {n: baseline.get(n, "") for n in settings}, "outcome": baseline_outcome}}
    if not candidates:
        result["message"] = "No observed configuration has strictly lower Performance-."
        return result
    count, outcome, record_id, row, changed = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    result.update({"repair": {"record_id": record_id, "configuration": {n: row.get(n, "") for n in settings}, "outcome": outcome,
                   "changed_settings": changed, "change_count": count,
                   "improvement": baseline_outcome - outcome},
                   "change_count_definition": "number of differing raw encoder-setting columns; correlated or one-hot fields may represent fewer conceptual changes",
                   "selection_priority": "strict improvement, then raw-column change count, Performance-, CSV record order",
                   "limitations": ["Observed improvements are not causal and do not guarantee future performance.", "This is an exhaustive retrospective comparison of finite observed records."]})
    return result

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        print("usage: x264_fix.py DATASET [BASELINE_RECORD_ID]", file=sys.stderr)
        return 2
    try:
        baseline_id = int(argv[1]) if len(argv) == 2 else 0
        print(json.dumps(fix(argv[0], baseline_id), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
