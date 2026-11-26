"""Aggregate saved per-step metrics into report-ready machine-readable tables.

This script only copies and derives values already recorded in step metrics. It
does not invent quality, transfer, or `same` results.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REQUIRED = {
    "step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc",
    "test_pass", "runtime_seconds", "error_count", "notes",
}


def read_metrics():
    rows = []
    for path in sorted((ROOT / "experiments/runs").glob("treatment[1-7]/trials/*/*/step*/metrics.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not REQUIRED.issubset(reader.fieldnames or set()):
                continue
            for row in reader:
                row = dict(row)
                row["source_metrics"] = str(path.relative_to(ROOT))
                row["new_code_percent"] = ""
                try:
                    total = float(row["total_loc"])
                    new = float(row["new_loc"])
                    if total:
                        row["new_code_percent"] = f"{100 * new / total:.6f}"
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
                rows.append(row)
    return rows


def write(name, rows, fields):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = read_metrics()
    common = ["step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc", "new_code_percent", "runtime_seconds", "error_count", "test_pass", "notes", "source_metrics"]
    write("metrics.csv", rows, common)
    write("code_size.csv", rows, ["treatment", "trial", "domain", "step", "skill", "total_loc", "new_loc", "new_code_percent", "source_metrics"])
    write("reliability.csv", rows, ["treatment", "trial", "domain", "step", "skill", "test_pass", "error_count", "runtime_seconds", "source_metrics"])
    print(f"Wrote {len(rows)} measured step rows")
    print("Quality, transfer, and same-comparison tables remain pending because no source measurements were supplied.")


if __name__ == "__main__":
    main()
