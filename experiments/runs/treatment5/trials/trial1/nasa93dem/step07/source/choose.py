#!/usr/bin/env python3
"""Retrospective, observed-outcome project selection under an effort budget."""
import csv
import math
import sys

DESCRIPTORS = ("prec", "flex", "resl", "team", "pmat",
               "rely", "data", "cplx", "ruse", "docu", "time", "stor",
               "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced")
# centerX and YearX are X-suffix metadata, kept out of the 22 descriptor settings.

def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(label + " must be numeric")
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number

def read_records(path):
    """Read rows while retaining their one-based original CSV record position."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"idX", "Kloc+", "Effort-"} | set(DESCRIPTORS)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("missing columns: " + ", ".join(missing))
        records = []
        for position, row in enumerate(reader, 1):
            records.append({"position": position, "row": row})
        return records

def choose(records, budget):
    budget = _finite(budget, "budget")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    feasible = []
    for record in records:
        row = record["row"]
        try:
            effort = _finite(row["Effort-"], "Effort-")
            size = _finite(row["Kloc+"], "Kloc+")
        except ValueError:
            continue
        if effort <= budget:
            feasible.append((size, effort, record["position"], record))
    if not feasible:
        return {"budget": budget, "total": len(records), "feasible": 0, "selected": None}
    # max delivered-size proxy; then lower effort; then earliest original CSV position.
    selected = max(feasible, key=lambda item: (item[0], -item[1], -item[2]))[3]
    return {"budget": budget, "total": len(records), "feasible": len(feasible), "selected": selected}

def report(result):
    selected = result["selected"]
    lines = ["Selection uses observed outcomes, not predictions for unseen proposals.",
             "Kloc+ is used only as an explicitly labeled delivered-size proxy for value; it is not code productivity or business value.",
             "budget=" + str(result["budget"]),
             "feasible/total=" + str(result["feasible"]) + "/" + str(result["total"]),
             "selection_rule=max observed Kloc+; ties=min observed Effort-; then earliest CSV record"]
    if selected is None:
        lines.append("selected=none (no feasible records with finite outcomes)")
        return "\n".join(lines)
    row = selected["row"]
    lines += ["original_record_position=" + str(selected["position"]),
              "original_record_identifier=" + row["idX"],
              "descriptor_settings=" + ", ".join(name + "=" + row[name] for name in DESCRIPTORS),
              "observed_effort=" + row["Effort-"],
              "delivered_size_proxy_Kloc+=" + row["Kloc+"]]
    return "\n".join(lines)

def main(argv):
    if len(argv) != 3:
        raise ValueError("usage: choose.py DATASET.csv BUDGET")
    print(report(choose(read_records(argv[1]), argv[2])))

if __name__ == "__main__":
    try:
        main(sys.argv)
    except (OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        sys.exit(2)
