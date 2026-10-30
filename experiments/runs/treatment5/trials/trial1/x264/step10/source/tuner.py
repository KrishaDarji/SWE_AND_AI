"""Independent EZR-informed race and uniform random search."""
import csv, math, random
TIE_TOLERANCE=0.0

def _validate_budget(budget,n):
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<1 or budget>n: raise ValueError("budget must be a positive integer")
def _prepare(candidates):
    out=[]; seen=set()
    for c in candidates:
        if set(c)!={"id","config"}: raise ValueError("candidate contains outcomes or invalid fields")
        cfg=tuple(c["config"])
        if cfg not in seen: seen.add(cfg); out.append({"id":c["id"],"config":cfg})
    return out
def _evaluate(c,callback):
    v=callback(c)
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v): raise ValueError("callback observation must be finite numeric")
    return float(v)
def _finish(cs,order,records):
    pos=min(range(len(records)),key=lambda p:(records[p][1],p)); idx,v=records[pos]
    return {"order":[cs[i]["id"] for i in order],"budget":len(order),"recommended":cs[idx],"measured":v}
def _norm(v,lo,hi): return 0.0 if lo==hi else (v-lo)/(hi-lo)
def race(candidates,budget,seed,callback):
    cs=_prepare(candidates); _validate_budget(budget,len(cs)); rng=random.Random(seed); order=[]; records=[]; seen=set()
    def pick(i):
        if i in seen: raise RuntimeError("duplicate evaluation")
        seen.add(i); order.append(i); records.append((i,_evaluate(cs[i],callback)))
    pick(rng.randrange(len(cs)))
    while len(order)<budget:
        ranges=[]
        for j in range(len(cs[0]["config"])):
            vals=[c["config"][j] for c in cs]; ranges.append((min(vals),max(vals)))
        def score(i):
            return min(sum((_norm(a,lo,hi)-_norm(b,lo,hi))**2 for a,b,(lo,hi) in zip(cs[i]["config"],cs[k]["config"],ranges)) for k in order)
        pick(max((i for i in range(len(cs)) if i not in seen),key=lambda i:(score(i),-i)))
    return _finish(cs,order,records)
def random_search(candidates,budget,seed,callback):
    cs=_prepare(candidates); _validate_budget(budget,len(cs)); order=list(range(len(cs))); random.Random(seed).shuffle(order); order=order[:budget]
    return _finish(cs,order,[(i,_evaluate(cs[i],callback)) for i in order])
def summarize(pairs,tolerance=TIE_TOLERANCE):
    if tolerance<0 or not math.isfinite(tolerance) or not pairs or not all(math.isfinite(float(x)) for x in pairs): raise ValueError("invalid summary")
    d=[float(x) for x in pairs]; s=sorted(d); n=len(s); m=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    w=sum(x < -tolerance for x in d); t=sum(abs(x)<=tolerance for x in d)
    return {"count":n,"wins":w,"ties":t,"losses":n-w-t,"mean":sum(d)/n,"median":m,"differences":d,"tolerance":tolerance}
def compare(path,budget=20,seeds=range(42,52),tolerance=TIE_TOLERANCE):
    with open(path,newline="") as f: rows=list(csv.DictReader(f))
    cols=[c for c in rows[0] if c!="Performance-"]; cs=[]; outcomes={}; configs=set()
    for i,row in enumerate(rows):
        cfg=tuple(int(row[c]) for c in cols)
        if cfg not in configs: configs.add(cfg); ident=i+1; cs.append({"id":ident,"config":cfg}); outcomes[ident]=float(row["Performance-"])
    results=[]
    for seed in seeds:
        for name,fn in (("race",race),("random",random_search)):
            calls=[]
            def callback(c,calls=calls):
                if set(c)!={"id","config"}: raise AssertionError("outcome leaked")
                calls.append(c["id"]); return outcomes[c["id"]]
            r=fn(cs,budget,seed,callback); r.update(strategy=name,seed=seed,columns=cols,calls=calls); results.append(r)
    paired=[next(x for x in results if x["seed"]==s and x["strategy"]=="race")["measured"]-next(x for x in results if x["seed"]==s and x["strategy"]=="random")["measured"] for s in seeds]
    return results,summarize(paired,tolerance)
