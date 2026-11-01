#!/usr/bin/env python3
"""Suggest the smallest improving change among observed configurations."""
import csv, math, sys
from summarize_runs import CONFIG, OUTCOMES, MISSING, finite_number

NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}

def _setting(row, name):
    value = row.get(name, "")
    if name == "criterion":
        value = value.strip().lower()
        return None if value in MISSING else value
    return finite_number(value)

def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try: header = [x.strip() for x in next(reader)]
        except StopIteration: return [], [], 0
        rows, malformed = [], 0
        for position, raw in enumerate(reader, 2):
            if len(raw) != len(header):
                malformed += 1
                rows.append((position, None))
                continue
            rows.append((position, {header[i]: raw[i].strip() for i in range(len(header))}))
        return header, rows, malformed

def _complete(row):
    config = {name: _setting(row, name) for name in CONFIG}
    outcomes = {name: finite_number(row.get(name, "")) for name in OUTCOMES}
    return config, outcomes, all(x is not None for x in config.values()) and all(x is not None for x in outcomes.values())

def suggest_fix(path):
    header, rows, malformed = _read(path)
    if not rows:
        return {"status": "empty", "reason": "no data rows after the header", "header": header, "malformed_count": malformed}
    baseline_config, baseline_outcomes, baseline_ok = _complete(rows[0][1] or {})
    base = {"position": rows[0][0], "configuration": baseline_config, "outcomes": baseline_outcomes}
    if not baseline_ok:
        return {"status": "invalid_baseline", "reason": "the first original data row lacks a complete finite configuration or finite outcomes", "baseline": base, "malformed_count": malformed, "candidate_count": 0, "improving_count": 0, "excluded_count": max(0, len(rows)-1)}
    candidates, excluded, improving = [], 0, 0
    for position, row in rows[1:]:
        if row is None: excluded += 1; continue
        config, outcomes, ok = _complete(row)
        if not ok: excluded += 1; continue
        if not outcomes["MRE-"] < baseline_outcomes["MRE-"]: continue
        improving += 1
        changed = [name for name in CONFIG if config[name] != baseline_config[name]]
        candidates.append((len(changed), outcomes["MRE-"], -outcomes["ACC+"], -outcomes["PRED40+"], position, config, outcomes, changed))
    if not candidates:
        return {"status": "no_improvement", "baseline": base, "malformed_count": malformed, "candidate_count": len(rows)-1, "improving_count": improving, "excluded_count": excluded, "selection_rule": "fewest changed fields, lowest MRE-, highest ACC+, highest PRED40+, earliest original CSV position"}
    chosen = min(candidates)
    _, _, _, _, position, config, outcomes, changed = chosen
    fields = [{"field": name, "old": baseline_config[name], "new": config[name]} for name in changed]
    return {"status": "ok", "baseline": base, "selected": {"position": position, "configuration": config, "outcomes": outcomes, "changed_fields": fields, "change_count": len(fields)}, "mre_improvement": baseline_outcomes["MRE-"] - outcomes["MRE-"], "secondary_differences": {name: outcomes[name] - baseline_outcomes[name] for name in ("ACC+", "PRED40+")}, "candidate_count": len(rows)-1, "improving_count": improving, "excluded_count": excluded, "malformed_count": malformed, "selection_rule": "fewest changed fields, lowest MRE-, highest ACC+, highest PRED40+, earliest original CSV position", "retrospective": "This compares observed historical rows. Change count is a simplicity proxy; differences do not prove causation or guarantee future improvement."}

def render(result):
    lines = ["Smallest observed improving hyperparameter change", "Status: " + result["status"]]
    if result.get("reason"): lines.append("Reason: " + result["reason"])
    if result.get("baseline"): lines.append("Baseline: " + str(result["baseline"]))
    if result.get("selected"):
        lines += ["Selected: " + str(result["selected"]), "Absolute MRE improvement: " + str(result["mre_improvement"]), "Secondary differences (selected minus baseline): " + str(result["secondary_differences"])]
    lines += ["Counts: candidates={}, improving={}, excluded={}, malformed={}".format(result.get("candidate_count", 0), result.get("improving_count", 0), result.get("excluded_count", 0), result.get("malformed_count", 0)), "Selection rule: " + result.get("selection_rule", "none"), "First data row is always the baseline; invalid or malformed first rows are reported, never replaced. Complete settings and finite outcomes are required; no compute cap is applied.", result.get("retrospective", "")]
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: fix.py DATASET.csv")
    print(render(suggest_fix(sys.argv[1])))
