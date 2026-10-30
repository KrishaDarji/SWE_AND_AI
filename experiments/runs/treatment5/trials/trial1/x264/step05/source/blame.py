#!/usr/bin/env python3
"""Rank encoder settings by observational association with Performance-."""
import argparse, csv, math, os, sys


def finite_number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


class Num:
    """EZR-style one-pass numeric summary (mean, M2, standard deviation)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def sd(self):
        return math.sqrt(max(0.0, self.m2) / (self.n - 1)) if self.n > 1 else 0.0


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no data rows")
    target = "Performance-"
    if target not in rows[0]:
        raise ValueError("missing Performance- column")
    settings = [name for name in rows[0] if name != target]
    usable = []
    for row in rows:
        outcome = finite_number(row.get(target))
        if outcome is not None:
            usable.append((row, outcome))
    return settings, usable


def summarize(values):
    s = Num()
    for value in values:
        s.add(value)
    return {"n": s.n, "mean": s.mean if s.n else None, "sd": s.sd}


def group_key(raw):
    value = finite_number(raw)
    if value is not None:
        return ("value", value)
    text = "" if raw is None else str(raw).strip()
    return ("missing",) if text in ("", "?", "NA", "nan", "NaN", "inf", "-inf") else ("value", text)


def rank_settings(settings, usable):
    outcome_summary = summarize(outcome for _, outcome in usable)
    total_m2 = (outcome_summary["sd"] ** 2) * max(0, len(usable) - 1)
    results = []
    for setting in settings:
        groups = {}
        for row, outcome in usable:
            groups.setdefault(group_key(row.get(setting)), []).append(outcome)
        between = 0.0
        summaries = []
        for key, outcomes in sorted(groups.items(), key=lambda item: repr(item[0])):
            summary = summarize(outcomes)
            between += len(outcomes) * (summary["mean"] - outcome_summary["mean"]) ** 2
            summaries.append((key, summary))
        score = between / total_m2 if total_m2 > 0 else 0.0
        means = [(key, summary["mean"], summary["n"]) for key, summary in summaries]
        best = min(means, key=lambda item: (item[1], repr(item[0]))) if means else None
        worst = max(means, key=lambda item: (item[1], repr(item[0]))) if means else None
        results.append({"setting": setting, "score": score, "groups": len(groups),
                        "summaries": summaries, "best": best, "worst": worst})
    results.sort(key=lambda item: (-item["score"], item["setting"]))
    return outcome_summary, results


def duplicate_settings(settings, usable):
    seen, duplicates = {}, []
    for setting in settings:
        signature = tuple(group_key(row.get(setting)) for row, _ in usable)
        if signature in seen:
            duplicates.append((seen[signature], setting))
        else:
            seen[signature] = setting
    return duplicates


def analyze(path):
    settings, usable = read_csv(path)
    overall, rankings = rank_settings(settings, usable)
    return settings, usable, overall, rankings, duplicate_settings(settings, usable)


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    settings, usable, overall, rankings, duplicates = analyze(args.dataset)
    print(f"Rows used: {len(usable)}")
    print(f"Performance-: n={overall['n']} mean={fmt(overall['mean'])} sd={fmt(overall['sd'])}")
    print("Ranking: explained variance (eta-squared), higher means stronger marginal association")
    for rank, item in enumerate(rankings, 1):
        direction = "undetermined"
        if item["best"] and item["worst"] and item["best"][0] != item["worst"][0]:
            direction = f"lower at {item['best'][0][-1]} than {item['worst'][0][-1]}"
        print(f"{rank}. {item['setting']}: score={item['score']:.6f} groups={item['groups']} direction={direction}")
        for key, summary in item["summaries"]:
            print(f"   group={key[-1]} n={summary['n']} mean={fmt(summary['mean'])} sd={fmt(summary['sd'])}")
    if duplicates:
        print("Redundant setting signatures: " + "; ".join(f"{a}={b}" for a, b in duplicates))
    else:
        print("Redundant setting signatures: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
