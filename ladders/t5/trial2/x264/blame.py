#!/usr/bin/env python3
"""Rank settings by marginal association with lower-is-better Performance-."""
import csv, math, statistics, sys

def finite(x):
    try:
        y=float(x); return y if math.isfinite(y) else None
    except (TypeError, ValueError): return None

def association(values, outcomes):
    pairs=[(v,y) for v,y in zip(values,outcomes) if finite(y) is not None and v not in ("", "?")]
    ys=[finite(y) for _,y in pairs]
    groups={}
    for v,y in zip((v for v,_ in pairs),ys): groups.setdefault(v,[]).append(y)
    counts={str(k):len(v) for k,v in groups.items()}
    summaries={str(k):{"n":len(v),"mean":statistics.fmean(v),"median":statistics.median(v),"min":min(v),"max":max(v)} for k,v in groups.items()}
    if len(groups)<2 or len(ys)<2: return 0.0, "constant/insufficient", counts, summaries, [], []
    mean=statistics.fmean(ys); total=sum((y-mean)**2 for y in ys)
    between=sum(len(v)*(statistics.fmean(v)-mean)**2 for v in groups.values())
    score=math.sqrt(between/total) if total else 0.0
    means={str(k):statistics.fmean(v) for k,v in groups.items()}
    low=min(means.values()); high=max(means.values())
    lowest=sorted(k for k,v in means.items() if v==low); highest=sorted(k for k,v in means.items() if v==high)
    numeric=all(finite(k) is not None for k in groups)
    if len(set(means.values()))==1: direction="tied"
    elif numeric:
        ordered=sorted((finite(k),means[str(k)]) for k in groups)
        increasing=all(a[1] <= b[1] for a,b in zip(ordered,ordered[1:]))
        decreasing=all(a[1] >= b[1] for a,b in zip(ordered,ordered[1:]))
        direction="increasing" if increasing else "decreasing" if decreasing else "nonmonotonic"
    else: direction="unordered-categorical"
    return score, direction, counts, summaries, lowest, highest

def analyze(path):
    with open(path, newline="") as f: rows=list(csv.DictReader(f))
    if not rows or "Performance-" not in rows[0]: raise ValueError("missing Performance- or empty CSV")
    settings=[k for k in rows[0] if k != "Performance-"]
    out=[]
    for name in settings:
        score,direction,counts,summaries,lowest,highest=association([r.get(name) for r in rows],[r.get("Performance-") for r in rows])
        means={k:v["mean"] for k,v in summaries.items()}
        out.append({"setting":name,"score":score,"direction":direction,"groups":len(counts),"counts":counts,"summaries":summaries,"lowest_mean_groups":lowest,"lowest_mean":min((means[k] for k in lowest),default=None),"highest_mean_groups":highest,"highest_mean":max((means[k] for k in highest),default=None)})
    out.sort(key=lambda x:(-x["score"],x["setting"]))
    return {"rows":len(rows),"outcome_valid":sum(finite(r.get("Performance-")) is not None for r in rows),"ranking":out}

def main():
    if len(sys.argv)!=2: raise SystemExit("usage: blame.py DATASET.csv")
    result=analyze(sys.argv[1]); print("rows",result["rows"],"valid_outcomes",result["outcome_valid"])
    print("setting,score,direction,groups")
    for x in result["ranking"]: print(f'{x["setting"]},{x["score"]:.6f},{x["direction"]},{x["groups"]},lowest={x["lowest_mean_groups"]}:{x["lowest_mean"]},highest={x["highest_mean_groups"]}:{x["highest_mean"]}')
    print("\nDetails:")
    for x in result["ranking"]: print(x["setting"],x["counts"],x["summaries"])
if __name__=="__main__": main()
