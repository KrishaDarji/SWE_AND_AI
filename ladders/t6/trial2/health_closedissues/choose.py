#!/usr/bin/env python3
"""Choose the best feasible observed historical configuration."""
import csv
import math
import sys
from summarize_runs import CONFIG, OUTCOMES, MISSING, finite_number

NUMERIC_CONFIG = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}

def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            return [], [], 0
        rows = []
        malformed = 0
        for position, raw in enumerate(reader, 2):
            if len(raw) != len(header):
                malformed += 1
                continue
            rows.append((position, {header[i]: raw[i].strip() for i in range(len(header))}))
        return header, rows, malformed

def _config(row):
    result = {}
    for name in CONFIG:
        value = row.get(name, "")
        if name == "criterion":
            result[name] = None if value.lower() in MISSING else value
        else:
            result[name] = finite_number(value)
    return result

def choose(path, cap=100):
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be a finite, nonnegative number")
    header, rows, malformed = _rows(path)
    raw_count = len(rows) + malformed
    valid = []; invalid = 0; over_budget = 0
    for position, row in rows:
        config = _config(row)
        outcomes = {name: finite_number(row.get(name, "")) for name in OUTCOMES}
        if (set(CONFIG) - set(row) or any(config[name] is None for name in CONFIG)
                or config["N_estimators"] < 0
                or any(value is None for value in outcomes.values())):
            invalid += 1
            continue
        if config["N_estimators"] > cap:
            over_budget += 1
            continue
        valid.append((position, config, outcomes))
    selected = min(valid, key=lambda item: (item[2]["MRE-"], -item[2]["ACC+"], -item[2]["PRED40+"], item[0])) if valid else None
    result = {"status": "ok" if selected else ("empty" if raw_count == 0 else "no_feasible_row"),
              "cap": cap, "raw_count": raw_count, "parsed_count": len(rows), "valid_count": len(rows)-invalid,
              "feasible_count": len(valid), "malformed_count": malformed, "invalid_count": invalid,
              "over_budget_count": over_budget,
              "exclusions": {"malformed_rows": malformed, "invalid_data": invalid, "valid_over_cap": over_budget},
              "header": header, "selection_rule": "lowest MRE-, then highest ACC+, then highest PRED40+, then earliest original CSV position",
              "feasibility": "complete finite configuration and finite outcomes; 0 <= N_estimators <= cap",
              "conventions": "Header is position 1; data positions are original CSV positions. Wrong-field-count rows are malformed and excluded; blank/missing/nonfinite fields are invalid.",
              "retrospective": "Selection uses observed historical outcomes and does not guarantee unseen performance; other settings may affect computation.",
              "selected": None}
    if selected:
        position, config, outcomes = selected
        result["selected"] = {"position": position, "configuration": config, "outcomes": outcomes, "proxy_value": config["N_estimators"]}
    return result

def render(result):
    lines = ["Retrospective feasible configuration selection", "Status: " + result["status"],
             "Counts: raw={}, parsed={}, valid={}, feasible={}".format(result["raw_count"], result["parsed_count"], result["valid_count"], result["feasible_count"]),
             "Exclusions: " + str(result["exclusions"]), "Cap: {} (selected proxy: N_estimators)".format(result["cap"]),
             "Selection rule: " + result["selection_rule"], "Feasibility: " + result["feasibility"]]
    if result["selected"] is not None:
        s = result["selected"]
        lines += ["Original position: {}".format(s["position"]), "Configuration: {}".format(s["configuration"]), "Observed outcomes: {}".format(s["outcomes"]), "Selected proxy value: {}".format(s["proxy_value"])]
    lines += [result["conventions"], result["retrospective"]]
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3): raise SystemExit("usage: choose.py DATASET.csv [cap]")
    try: cap = 100 if len(sys.argv) == 2 else float(sys.argv[2]); print(render(choose(sys.argv[1], cap)))
    except ValueError as error: raise SystemExit(str(error))
