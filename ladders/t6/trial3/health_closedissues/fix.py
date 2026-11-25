"""Smallest observed configuration change that strictly improves MRE-."""
import csv, math
from analyzer import CONFIGS, OUTCOMES, _number, _cat

def _settings(row):
    return {k: (_cat(row.get(k)) if t == "categorical" else _number(row.get(k))) for k,t in CONFIGS.items()}

def _complete(row):
    s, o = _settings(row), {k: _number(row.get(k)) for k in OUTCOMES}
    return all(v is not None for v in s.values()) and all(v is not None for v in o.values()), s, o

def _first_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f); header = next(rd, None)
        if header is None: return None, []
        header = [x.strip() for x in header]; rows=[]
        for pos, cells in enumerate(rd, 2):
            if not cells or all(not x.strip() for x in cells):
                continue
            if len(cells) == len(header): rows.append((pos, dict(zip(header, cells))))
            else: rows.append((pos, None))
        return header, rows

def _changed(old, new):
    return {k: {"old": old[k], "new": new[k]} for k in CONFIGS if old[k] != new[k]}

def fix(path):
    header, physical = _first_rows(path)
    if header is None: return {"status":"empty_input", "message":"No header or data rows are available.", "baseline":None, "candidate":None}
    if not physical: return {"status":"empty_input", "message":"Header exists but no original CSV data row exists.", "baseline":None, "candidate":None}
    first_pos, first = physical[0]
    if first is None: return {"status":"invalid_baseline", "message":"The first original CSV data row is malformed; it was not replaced by a later row.", "baseline":{"original_position":first_pos}, "candidate":None}
    ok, base_s, base_o = _complete(first)
    if not ok:
        return {"status":"invalid_baseline", "message":"The first original CSV data row lacks complete finite settings or outcomes; it was not replaced by a later row.", "baseline":{"original_position":first_pos, "configuration":base_s, "outcomes":base_o}, "candidate":None}
    eligible=0; excluded=0; improving=[]
    for pos,row in physical[1:]:
        if row is None: excluded += 1; continue
        ok,s,o=_complete(row)
        if not ok: excluded += 1; continue
        eligible += 1
        if o["MRE-"] < base_o["MRE-"]:
            changes=_changed(base_s,s)
            improving.append((len(changes),o["MRE-"],-o["ACC+"],-o["PRED40+"],pos,s,o,changes))
    result={"status":"ok", "baseline":{"original_position":first_pos,"configuration":base_s,"outcomes":base_o},
            "candidate":None, "candidate_count":len(improving), "eligible_candidate_rows":eligible,
            "excluded_candidate_rows":excluded, "selection_rule":["fewest changed configuration fields","lowest MRE-","highest ACC+","highest PRED40+","earliest original CSV position"],
            "interpretation":"Retrospective comparison between observed rows. Change count is a simplicity proxy; historical differences do not prove causation or guarantee future improvement."}
    if not improving:
        result["status"]="no_improving_candidate"; result["message"]="No complete observed row has strictly lower MRE- than the baseline."; return result
    _,_,_,_,pos,s,o,ch=min(improving)
    result["candidate"]={"original_position":pos,"configuration":s,"outcomes":o,"changed_fields":ch,"change_count":len(ch),
        "absolute_mre_improvement":base_o["MRE-"]-o["MRE-"],
        "secondary_outcome_differences":{k:o[k]-base_o[k] for k in ("ACC+","PRED40+")},
        "secondary_outcome_worsening":{k:o[k]<base_o[k] for k in ("ACC+","PRED40+")}}
    return result
