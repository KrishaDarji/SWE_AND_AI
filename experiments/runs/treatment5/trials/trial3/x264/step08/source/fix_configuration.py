#!/usr/bin/env python3
"""Find the least-changing observed configuration with lower Performance-."""

import csv
import math
from pathlib import Path


OUTCOME = "Performance-"


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def read_records(path):
    """Read CSV rows, retaining raw strings and assigning 1-based record IDs."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0] or OUTCOME not in rows[0]:
        raise ValueError("CSV must contain at least one row and Performance-")
    settings = [name for name in rows[0] if name != OUTCOME]
    return [dict(record_id=i, settings={k: row.get(k) for k in settings},
                 outcome=row.get(OUTCOME)) for i, row in enumerate(rows, 1)], settings


def repair(baseline, records, setting_names):
    """Return a report, or an explicit no-repair report."""
    if baseline is None:
        raise ValueError("baseline is required")
    baseline_outcome = _number(baseline.get("outcome"), "baseline outcome")
    for name in setting_names:
        _number(baseline["settings"].get(name), f"baseline setting {name}")

    choices = []
    for candidate in records:
        try:
            outcome = _number(candidate.get("outcome"),
                             f"record {candidate.get('record_id')} outcome")
            changes = []
            for name in setting_names:
                old = _number(baseline["settings"].get(name), f"baseline setting {name}")
                new = _number(candidate["settings"].get(name),
                              f"record {candidate.get('record_id')} setting {name}")
                if old != new:
                    changes.append({"setting": name, "old": baseline["settings"][name],
                                    "new": candidate["settings"][name]})
        except ValueError:
            continue
        if outcome < baseline_outcome:
            choices.append((len(changes), outcome, candidate["record_id"], candidate, changes))
    if not choices:
        return {"repaired": False, "reason": "no observed configuration has strictly lower Performance-",
                "baseline": baseline, "baseline_outcome": baseline_outcome}
    _, outcome, _, candidate, changes = min(choices, key=lambda item: item[:3])
    return {"repaired": True, "baseline": baseline, "repaired_record": candidate,
            "baseline_outcome": baseline_outcome, "repaired_outcome": outcome,
            "changed_settings": changes, "change_count": len(changes),
            "improvement": baseline_outcome - outcome,
            "change_count_note": "Raw-column count is a proxy; correlated or one-hot fields may represent fewer conceptual changes."}


def main(dataset):
    records, settings = read_records(dataset)
    baseline = records[0]
    report = repair(baseline, records, settings)
    report["identifier_definition"] = "record_id is the 1-based data-record order after the CSV header."
    report["limitations"] = ["Observed improvement is not causal and does not guarantee future performance.",
                             "Only configurations present in this dataset are considered."]
    return report


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(main(args.dataset), indent=2, sort_keys=True))
