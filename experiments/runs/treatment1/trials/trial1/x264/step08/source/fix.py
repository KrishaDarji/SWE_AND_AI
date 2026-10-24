"""Suggest the smallest one-setting change that improves an X264 configuration."""
import argparse
import csv
import json
import sys


def read_measurements(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no measurements")
    settings = [name for name in rows[0] if name != "Performance-"]
    if not settings:
        raise ValueError("dataset has no setting columns")
    measurements = []
    for row in rows:
        try:
            config = {name: int(row[name].strip()) for name in settings}
            score = float(row["Performance-"].strip())
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid measurement row") from exc
        if any(value not in (0, 1) for value in config.values()):
            raise ValueError("settings must be binary")
        measurements.append((config, score))
    return settings, measurements


def suggest_fix(path, current):
    settings, measurements = read_measurements(path)
    unknown = set(current) - set(settings)
    missing = set(settings) - set(current)
    if unknown or missing:
        raise ValueError("configuration must contain exactly the dataset settings")
    if any(current[name] not in (0, 1) for name in settings):
        raise ValueError("configuration values must be 0 or 1")
    baseline = [score for config, score in measurements if config == current]
    if not baseline:
        raise ValueError("current configuration is not measured")
    baseline_score = sum(baseline) / len(baseline)
    candidates = []
    for name in settings:
        changed = dict(current)
        changed[name] = 1 - changed[name]
        scores = [score for config, score in measurements if config == changed]
        if scores:
            score = sum(scores) / len(scores)
            if score > baseline_score:
                candidates.append((score - baseline_score, name, changed, score))
    if not candidates:
        return {"improvement": False, "baseline_score": baseline_score,
                "message": "no measured one-setting change improves this configuration"}
    gain, name, changed, score = max(candidates, key=lambda item: (item[0], item[1]))
    return {"improvement": True, "setting": name, "from": current[name],
            "to": changed[name], "baseline_score": baseline_score,
            "suggested_score": score, "gain": gain, "configuration": changed}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("configuration", help="JSON object mapping setting names to 0 or 1")
    args = parser.parse_args(argv)
    try:
        result = suggest_fix(args.dataset, json.loads(args.configuration))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
