"""Robust marginal screening of unusual model-quality outcomes.

Scores are absolute deviations from the outcome median divided by
1.4826 * MAD.  The assessed record is included in its outcome reference.
Scores at least 3.5 are investigation candidates; this is not an error
detector and does not condition on configuration settings.
"""
import csv
import json
import math
import statistics
from pathlib import Path

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
THRESHOLD = 3.5
MAD_SCALE = 1.4826


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _record_id(row, index):
    return row.get("record_id", index)


def screen(rows, threshold=THRESHOLD):
    """Return JSON-safe screening results for an iterable of mappings."""
    rows = list(rows)
    references = {}
    for field in OUTCOMES:
        values = [_number(row.get(field)) for row in rows]
        valid = [value for value in values if value is not None]
        median = statistics.median(valid) if valid else None
        mad = statistics.median([abs(value - median) for value in valid]) if valid else None
        scale = MAD_SCALE * mad if mad is not None else None
        references[field] = {"valid_count": len(valid), "median": median, "mad": mad,
                             "scale": scale, "includes_assessed_record": True}
    records = []
    field_counts = {field: 0 for field in OUTCOMES}
    flagged_ids = set()
    for index, row in enumerate(rows):
        outcomes = {}
        record_id = _record_id(row, index)
        for field in OUTCOMES:
            observed = _number(row.get(field))
            ref = references[field]
            score = None
            flagged = False
            if observed is None:
                reason = "observed value is missing or non-finite"
            elif ref["valid_count"] < 2:
                reason = "undefined: fewer than two finite reference values"
            elif ref["scale"] == 0:
                if observed == ref["median"]:
                    score, reason = 0.0, "zero robust scale; observed equals reference median"
                else:
                    reason = "undefined/unbounded: zero robust scale and non-median observation"
            else:
                score = abs(observed - ref["median"]) / ref["scale"]
                flagged = score >= threshold
                reason = "score meets threshold" if flagged else "defined score below threshold"
            if flagged:
                field_counts[field] += 1
                flagged_ids.add(record_id)
            outcomes[field] = {"observed": observed, "reference_median": ref["median"],
                               "score": score, "threshold": threshold, "flagged": flagged,
                               "reason": reason}
        records.append({"record_id": record_id, "outcomes": outcomes})
    return {"method": "median/MAD", "threshold": threshold,
            "reference_includes_assessed_record": True, "references": references,
            "records": records, "flagged_record_count": len(flagged_ids),
            "flag_count_by_outcome": field_counts,
            "interpretation": "Flags are investigation candidates. This marginal screen does not condition on configuration settings and may flag legitimate configuration differences; no detection accuracy is claimed without verified labels."}


def screen_csv(path, threshold=THRESHOLD):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return screen(csv.DictReader(handle), threshold)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(screen_csv(args.dataset), allow_nan=False, indent=2, sort_keys=True))
