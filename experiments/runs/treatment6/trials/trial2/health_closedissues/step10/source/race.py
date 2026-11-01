#!/usr/bin/env python3
"""Paired retrospective benchmark of choose versus uniform historical sampling."""
import math, random, statistics, sys
from choose import _rows, _config
from summarize_runs import CONFIG, OUTCOMES, finite_number, MISSING

def _check(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite, nonnegative number")

def eligible(path, cap=100):
    _check(cap, "cap")
    header, rows, malformed = _rows(path); valid=[]; invalid=over=0
    for position, row in rows:
        config=_config(row); outcomes={n: finite_number(row.get(n, "")) for n in OUTCOMES}
        if set(CONFIG)-set(row) or any(config[n] is None for n in CONFIG) or config["N_estimators"] < 0 or any(v is None for v in outcomes.values()):
            invalid += 1; continue
        if config["N_estimators"] > cap: over += 1; continue
        valid.append({"position":position, "configuration":config, "outcomes":outcomes})
    return {"header":header,"raw_count":len(rows)+malformed,"parsed_count":len(rows),"malformed_count":malformed,"invalid_count":invalid,"over_budget_count":over,"valid_count":len(rows)-invalid,"feasible_count":len(valid),"exclusions":{"malformed_rows":malformed,"invalid_data":invalid,"valid_over_cap":over},"rows":valid}

def _selected(rows):
    return min(rows, key=lambda x:(x["outcomes"]["MRE-"],-x["outcomes"]["ACC+"],-x["outcomes"]["PRED40+"],x["position"])) if rows else None

def classify(difference, tolerance):
    _check(tolerance, "tolerance")
    if difference > tolerance: return "win"
    if difference < -tolerance: return "loss"
    return "tie"

def race(path, cap=100, seeds=range(42,52), tolerance=1e-12):
    _check(tolerance, "tolerance"); pop=eligible(path,cap); rows=pop["rows"]
    deterministic=_selected(rows); results=[]
    for seed in seeds:
        rng=random.Random(seed); baseline=rng.choice(rows) if rows else None
        diff=None if not baseline else baseline["outcomes"]["MRE-"]-deterministic["outcomes"]["MRE-"]
        results.append({"seed":seed,"selector":deterministic,"baseline":baseline,"mre_difference":diff,"classification":None if diff is None else classify(diff,tolerance),"acc_difference":None if not baseline else baseline["outcomes"]["ACC+"]-deterministic["outcomes"]["ACC+"],"pred40_difference":None if not baseline else baseline["outcomes"]["PRED40+"]-deterministic["outcomes"]["PRED40+"]})
    diffs=[x["mre_difference"] for x in results if x["mre_difference"] is not None]; counts={k:sum(x["classification"]==k for x in results) for k in ("win","tie","loss")}
    return {"status":"empty" if pop["raw_count"]==0 else "no_feasible_row" if not rows else "ok","cap":cap,"tolerance":tolerance,"population":{k:v for k,v in pop.items() if k!="rows"},"selection_rule":"lowest MRE-, then highest ACC+, then highest PRED40+, then earliest original CSV position","baseline_rule":"one eligible historical row uniformly at random using fresh random.Random(seed)","results":results,"summary":{"mean_difference":statistics.mean(diffs) if diffs else None,"median_difference":statistics.median(diffs) if diffs else None,**counts}}

def render(a):
    lines=["Selector versus historical baseline",f"Status: {a['status']}",f"Counts: {a['population']}",f"Exclusions: {a['population']['exclusions']}",f"Cap: {a['cap']}; tie tolerance: {a['tolerance']}",f"Selector rule: {a['selection_rule']}",f"Baseline rule: {a['baseline_rule']}","Primary difference: baseline MRE- minus selector MRE-; positive favors selector."]
    for x in a["results"]: lines.append(f"Seed {x['seed']}: selector={x['selector']}; baseline={x['baseline']}; MRE difference={x['mre_difference']} ({x['classification']}); ACC+ difference baseline-selector={x['acc_difference']}; PRED40+ difference baseline-selector={x['pred40_difference']}")
    lines += [f"Summary: {a['summary']}","The selector inspects historical outcomes across feasible rows; the baseline samples one row. This is retrospective benchmarking, not an equal-evaluation-budget search or proof of future superiority. N_estimators is a compute proxy, not measured runtime or evaluation cost.","Seeds repeat baseline selections from one historical dataset, not independent LLM trials or new datasets; no significance, confidence, or generalization guarantee is inferred."]
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) not in (2,3): raise SystemExit("usage: race.py DATASET.csv [cap]")
    try: print(render(race(sys.argv[1],100 if len(sys.argv)==2 else float(sys.argv[2]))))
    except ValueError as e: raise SystemExit(str(e))
