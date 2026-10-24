"""Find the smallest one-setting change that improves issue-closure health."""
import argparse
import csv
import json
import sys

METRICS = ("MRE-", "ACC+", "PRED40+")


def _number(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_dataset(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no rows")
    required = set(METRICS)
    if not required.issubset(rows[0]):
        raise ValueError("dataset must contain MRE-, ACC+, and PRED40+")
    settings = [name for name in rows[0] if name not in METRICS]
    if not settings:
        raise ValueError("dataset has no changeable settings")
    parsed = []
    for raw in rows:
        try:
            config = {name: _number(raw[name].strip()) for name in settings}
            metrics = {name: float(raw[name].strip()) for name in METRICS}
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid dataset row") from exc
        parsed.append((config, metrics))
    return settings, parsed


def suggest_fix(path, baseline=None):
    settings, rows = read_dataset(path)
    if baseline is None:
        baseline, base_metrics = max(rows, key=lambda item: item[1]["MRE-"])
    else:
        if set(baseline) != set(settings):
            raise ValueError("baseline must contain exactly the dataset settings")
        matches = [(config, metrics) for config, metrics in rows if config == baseline]
        if not matches:
            raise ValueError("baseline configuration is not measured")
        base_metrics = min((metrics for _, metrics in matches),
                           key=lambda item: item["MRE-"])

    candidates = []
    for setting in settings:
        values = sorted({config[setting] for config, _ in rows}, key=str)
        for value in values:
            if value == baseline[setting]:
                continue
            changed = dict(baseline)
            changed[setting] = value
            matches = [metrics for config, metrics in rows if config == changed]
            if not matches:
                continue
            metrics = min(matches, key=lambda item: (item["MRE-"], -item["ACC+"], -item["PRED40+"]))
            if metrics["MRE-"] < base_metrics["MRE-"]:
                distance = abs(float(value) - float(baseline[setting])) if isinstance(value, (int, float)) else 1
                candidates.append((metrics["MRE-"], distance, setting, str(value), changed, metrics))
    if not candidates:
        return {"improvement": False, "baseline": baseline, "baseline_metrics": base_metrics,
                "message": "no measured one-setting change improves issue-closure health"}
    _, distance, setting, _, changed, metrics = min(
        candidates, key=lambda item: (item[0], item[1], item[2]))
    return {"improvement": True, "setting": setting, "from": baseline[setting],
            "to": changed[setting], "baseline": baseline, "suggested": changed,
            "baseline_metrics": base_metrics, "suggested_metrics": metrics,
            "mre_reduction": base_metrics["MRE-"] - metrics["MRE-"],
            "change_distance": distance}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--baseline", help="JSON configuration; defaults to the highest-MRE row")
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline) if args.baseline else None
        print(json.dumps(suggest_fix(args.dataset, baseline), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
