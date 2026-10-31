"""Choose the highest observed Kloc+ project within an Effort- budget."""
import csv
import math
import sys

DESCRIPTORS = ("prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
               "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
               "apex", "plex", "ltex", "tool", "site", "sced")
REQUIRED = ("idX", "centerX", "YearX") + DESCRIPTORS + ("Kloc+", "Effort-")


class SelectionError(ValueError):
    """Raised for invalid selection inputs or unusable records."""


def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError("invalid {}: {!r}".format(label, value)) from exc
    if not math.isfinite(number):
        raise SelectionError("nonfinite {}: {!r}".format(label, value))
    return number


def choose(records, budget):
    """Return a report; records are mappings and retain their input position."""
    budget = _finite(budget, "budget")
    if budget < 0:
        raise SelectionError("budget must be nonnegative")
    total = len(records)
    feasible = []
    for position, record in enumerate(records, 1):
        effort = _finite(record.get("Effort-"), "Effort-")
        size = _finite(record.get("Kloc+"), "Kloc+")
        if effort >= 0 and effort <= budget:
            feasible.append((size, effort, position, record))
    if not feasible:
        raise SelectionError("no feasible records")
    size, effort, position, record = min(feasible, key=lambda item: (-item[0], item[1], item[2]))
    return {"budget": budget, "feasible_count": len(feasible), "total_count": total,
            "record_position": position, "record_id": record.get("idX"),
            "descriptors": {name: record.get(name) for name in DESCRIPTORS},
            "observed_effort": effort, "delivered_size": size,
            "selection_rule": "max observed Kloc+; tie: lower observed Effort-; then earliest CSV record"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise SelectionError("missing columns: " + ", ".join(missing))
        return list(reader)


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: choose.py DATASET BUDGET")
    report = choose(read_csv(argv[1]), argv[2])
    print("Observed selection (not a prediction for unseen proposals)")
    print("budget={}, feasible/total={}/{}, record_position={}, original_id={!r}".format(
        report["budget"], report["feasible_count"], report["total_count"],
        report["record_position"], report["record_id"]))
    print("observed_effort={}, delivered_size={}".format(report["observed_effort"], report["delivered_size"]))
    print("descriptor_settings=" + repr(report["descriptors"]))
    print("selection_rule=" + report["selection_rule"])
    print("Kloc+ is a delivered-size proxy, not business value or productivity.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except SelectionError as error:
        print("error: " + str(error), file=sys.stderr)
        raise SystemExit(2)
