"""Validated descriptive summaries of historical encoder measurements."""

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


OBJECTIVE = "Performance-"


@dataclass(frozen=True)
class Measurements:
    """Numeric records in CSV order; row numbers refer to data rows, starting at 1."""

    settings: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]


def load_measurements(path):
    """Read numeric settings and the registered objective without changing the CSV."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV must have nonempty column names")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate column names")
        if OBJECTIVE not in header:
            raise ValueError(f"Missing registered objective: {OBJECTIVE}")
        settings = tuple(name for name in header if name != OBJECTIVE)
        if not settings:
            raise ValueError("CSV must contain encoder settings")
        order = [header.index(name) for name in (*settings, OBJECTIVE)]
        records = []
        for line, cells in enumerate(reader, start=2):
            if len(cells) != len(header):
                raise ValueError(f"CSV line {line}: expected {len(header)} fields")
            try:
                values = tuple(float(cells[index]) for index in order)
            except ValueError as error:
                raise ValueError(f"CSV line {line}: missing or nonnumeric value") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"CSV line {line}: nonfinite value")
            records.append(values)
    if not records:
        raise ValueError("CSV has no measurement rows")
    return Measurements(settings, tuple(records))


def percentile(values, fraction):
    """Linearly interpolate the sorted sample at (n - 1) * fraction."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Percentile needs values and a fraction between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values):
    """Use population standard deviation for the complete supplied history."""
    return {
        "count": len(values),
        "min": min(values),
        "q25": percentile(values, 0.25),
        "median": statistics.median(values),
        "q75": percentile(values, 0.75),
        "max": max(values),
        "mean": statistics.mean(values),
        "population_stddev": statistics.pstdev(values),
    }


def summarize(data, top_n=5):
    """Return JSON-compatible facts; conditional statistics do not imply causation."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    objectives = [row[-1] for row in data.rows]
    setting_summary = {}
    for index, name in enumerate(data.settings):
        groups = {}
        for row in data.rows:
            groups.setdefault(row[index], []).append(row[-1])
        setting_summary[name] = {
            "constant": len(groups) == 1,
            "binary": set(groups).issubset({0.0, 1.0}),
            "values": [
                {
                    "value": value,
                    "count": len(group),
                    "fraction": len(group) / len(data.rows),
                    "objective": describe(group),
                }
                for value, group in sorted(groups.items())
            ],
        }
    ranked = sorted(enumerate(data.rows, start=1), key=lambda item: item[1][-1])

    def record(item):
        number, row = item
        return {
            "data_row": number,
            "settings": dict(zip(data.settings, row[:-1])),
            OBJECTIVE: row[-1],
        }

    configurations = {row[:-1] for row in data.rows}
    best_value = min(objectives)
    return {
        "row_count": len(data.rows),
        "setting_count": len(data.settings),
        "columns": [*data.settings, OBJECTIVE],
        "objective": {"name": OBJECTIVE, "direction": "minimize", "units": None,
                      "statistics": describe(objectives)},
        "quality_measures": [],
        "quality_note": "No separate quality measures are present in this dataset.",
        "unique_configurations": len(configurations),
        "repeated_configuration_rows": len(data.rows) - len(configurations),
        "duplicate_measurement_rows": len(data.rows) - len(set(data.rows)),
        "settings": setting_summary,
        "best_runs": [record(item) for item in ranked if item[1][-1] == best_value],
        "top_runs": [record(item) for item in ranked[:top_n]],
        "interpretation": [
            "All non-objective columns are treated as recorded encoder settings.",
            "Names and encoded values are preserved; flag semantics are not inferred.",
            "Conditional objective statistics describe associations, not causal effects.",
            "Repeated rows remain in all statistics; no measurements are imputed.",
            "Top runs are ordered by ascending objective, then original CSV order.",
        ],
    }


def render_text(summary):
    stats = summary["objective"]["statistics"]
    lines = [
        f"Historical encoder runs: {summary['row_count']}",
        f"Settings: {summary['setting_count']}",
        f"Unique configurations: {summary['unique_configurations']}",
        f"Repeated configuration rows: {summary['repeated_configuration_rows']}",
        f"Duplicate measurement rows: {summary['duplicate_measurement_rows']}",
        "Objective: Performance- (minimize; units unspecified)",
        "Objective statistics: " + json.dumps(stats, sort_keys=True),
        summary["quality_note"],
        "Setting distributions (value: count; mean Performance-):",
    ]
    for name, setting in summary["settings"].items():
        values = "; ".join(
            f"{item['value']:g}: {item['count']}; mean={item['objective']['mean']:.6f}"
            for item in setting["values"]
        )
        lines.append(f"  {name}{' [constant]' if setting['constant'] else ''}: {values}")
    lines.append(f"Best objective: {stats['min']}; tied runs: {len(summary['best_runs'])}")
    for run in summary["top_runs"]:
        lines.append("  " + json.dumps(run, sort_keys=True))
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to X264_AllMeasurements.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        summary = summarize(load_measurements(args.dataset), args.top)
        summary["dataset"] = str(args.dataset)
        report = render_text(summary)
        targets = [args.output_dir / name for name in ("summary.json", "summary.txt")]
        if any(target.resolve() == args.dataset.resolve() for target in targets):
            raise ValueError("Output must not overwrite the dataset")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        targets[0].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
        targets[1].write_text(report, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
