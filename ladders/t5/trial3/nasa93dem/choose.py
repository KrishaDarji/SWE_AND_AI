#!/usr/bin/env python3
"""Retrospectively choose the highest observed Kloc+ within an effort budget."""
import argparse
import csv
import math
import sys

# X-suffix fields are metadata. The 22 non-X fields are the descriptor settings.
DESCRIPTORS = ["prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
               "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
               "apex", "plex", "ltex", "tool", "site", "sced"]
OUTCOMES = ("Kloc+", "Effort-", "Defects-", "MonthsX")

def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def choose(path, budget):
    """Return a report dict; invalid rows remain counted in total records."""
    limit = finite(budget)
    if limit is None or limit < 0:
        raise ValueError("budget must be a finite nonnegative number")
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("input is empty")
        needed = ["idX"] + DESCRIPTORS + list(OUTCOMES)
        missing = [name for name in needed if name not in reader.fieldnames]
        if missing:
            raise ValueError("missing columns: " + ", ".join(missing))
        records = list(reader)
    if not records:
        raise ValueError("input has no data records")
    feasible = []
    for position, row in enumerate(records, 1):
        effort = finite(row.get("Effort-"))
        size = finite(row.get("Kloc+"))
        if effort is not None and size is not None and effort <= limit:
            feasible.append((size, effort, position, row))
    if not feasible:
        return {"budget": limit, "total": len(records), "feasible": 0, "selected": None}
    size, effort, position, row = min(feasible, key=lambda item: (-item[0], item[1], item[2]))
    return {"budget": limit, "total": len(records), "feasible": len(feasible),
            "selected": {"position": position, "idX": row["idX"],
                          "descriptors": {name: row[name] for name in DESCRIPTORS},
                          "effort": effort, "Kloc+": size}}

def render(report):
    lines = ["Retrospective observed-outcome selection", f"Budget: {report['budget']}",
             f"Feasible records: {report['feasible']}", f"Total records: {report['total']}",
             "Selection rule: maximize observed Kloc+; tie-break by lower observed Effort-; then earliest CSV record.",
             "Selection uses observed outcomes, not predictions for unseen proposals."]
    selected = report["selected"]
    if selected is None:
        lines.append("Selected record: none (no feasible record with finite outcomes).")
    else:
        lines += [f"Original CSV record position: {selected['position']}",
                  f"Original record identifier (idX; not assumed unique): {selected['idX']}",
                  "Descriptor settings (metadata and outcomes are separate):"]
        lines += [f"  {key}: {value}" for key, value in selected["descriptors"].items()]
        lines += [f"Observed effort: {selected['effort']}", f"Delivered-size proxy, observed Kloc+: {selected['Kloc+']}",
                  "Kloc+ is an explicitly labeled delivered-size proxy, not business value, cost, or productivity.",
                  "This observed selection does not claim settings cause outcomes or guarantee future feasibility."]
    return "\n".join(lines) + "\n"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--budget", type=float, default=100.0)
    args = parser.parse_args(argv)
    try:
        print(render(choose(args.csv_path, args.budget)), end="")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
