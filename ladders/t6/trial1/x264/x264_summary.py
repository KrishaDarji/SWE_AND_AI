"""Summarize X264 encoder settings and measured outcomes."""
import csv
import math
import sys


def _number(value):
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None


class NumericSummary:
    """One-pass mean and sample standard deviation, adapted from EZR Num."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def as_dict(self):
        return {"count": self.count, "mean": self.mean if self.count else None,
                "min": None, "max": None,
                "sample_stddev": math.sqrt(max(self.m2, 0.0) / (self.count - 1))
                if self.count > 1 else 0.0}


def summarize(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields:
            return {"record_count": 0, "settings": {}, "outcomes": {},
                    "missing_values": {}, "errors": ["missing header"]}
        outcome_names = [name for name in fields if name.endswith("+") or name.endswith("-")]
        outcome_names = [name for name in outcome_names if name == "Performance-" or name.endswith("-")]
        if not outcome_names and "Performance-" in fields:
            outcome_names = ["Performance-"]
        settings = {name: {} for name in fields if name not in outcome_names}
        outcomes = {name: NumericSummary() for name in outcome_names}
        missing = {name: 0 for name in fields}
        invalid = {name: 0 for name in outcome_names}
        records = 0
        for row in reader:
            records += 1
            for name in fields:
                value = (row.get(name) or "").strip()
                if value in ("", "?"):
                    missing[name] += 1
                elif name in settings:
                    settings[name][value] = settings[name].get(value, 0) + 1
            for name, summary in outcomes.items():
                value = (row.get(name) or "").strip()
                number = _number(value) if value not in ("", "?") else None
                if number is None and value not in ("", "?"):
                    invalid[name] += 1
                elif number is not None:
                    summary.add(number)
        result = {"record_count": records, "settings": settings,
                  "outcomes": {}, "missing_values": missing,
                  "outcome_identification": {"names": outcome_names,
                                              "direction": {n: "lower is better" for n in outcome_names}},
                  "invalid_outcomes": invalid,
                  "assumptions": ["CSV header supplies column names.",
                                  "Names ending in '-' are minimization outcomes; Performance- is the available measured outcome.",
                                  "Blank and '?' cells are missing; non-finite or non-numeric outcomes are invalid."]}
        for name, summary in outcomes.items():
            stats = summary.as_dict()
            values = []
            with open(path, newline="", encoding="utf-8-sig") as replay:
                for row in csv.DictReader(replay):
                    number = _number(row.get(name, ""))
                    if number is not None:
                        values.append(number)
            stats["min"] = min(values) if values else None
            stats["max"] = max(values) if values else None
            result["outcomes"][name] = stats
        return result


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("usage: x264_summary.py DATASET", file=sys.stderr)
        return 2
    import json
    print(json.dumps(summarize(argv[0]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
