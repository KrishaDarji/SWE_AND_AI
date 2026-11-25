"""Certification of the fixed historical configuration comparison."""
import math
try:
    from .analyzer import _number, _cat, _read, _summary
except ImportError:
    from analyzer import _number, _cat, _read, _summary
REFERENCE="squared_error"; CANDIDATE="absolute_error"; PRIMARY="MRE-"
def _betacf(a,b,x):
    qab,qap,qam=a+b,a+1,a-1; tiny=1e-300; c=1.; d=1-qab*x; d=tiny if abs(d)<tiny else d; d=1/d; h=d
    for m in range(1,10000):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; c=1+aa/c; d=tiny if abs(d)<tiny else d; c=tiny if abs(c)<tiny else c; d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; c=1+aa/c; d=tiny if abs(d)<tiny else d; c=tiny if abs(c)<tiny else c; d=1/d; delta=d*c; h*=delta
        if abs(delta-1)<3e-14: break
    return h
def _ibeta(a,b,x):
    if x<=0:return 0.
    if x>=1:return 1.
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x<(a+1)/(a+b+2): return bt*_betacf(a,b,x)/a
    y=1-x; bt2=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+b*math.log(y)+a*math.log1p(-y))
    return 1-bt2*_betacf(b,a,y)/b
def t_cdf(t,df):
    if not df>0 or not math.isfinite(t): return None
    if t==0:return .5
    if df==1: return .5+math.atan(t)/math.pi
    p=.5*_ibeta(df/2,.5,df/(df+t*t)); return 1-p if t>0 else p
def t_critical(df):
    lo,hi=0.,1.
    while t_cdf(hi,df)<.975: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<.975:lo=mid
        else:hi=mid
    return (lo+hi)/2
def _sample(v):
    s=_summary(v); return s["finite_count"],s["mean"],None if s["sample_sd"] is None else s["sample_sd"]**2
def _group(rows,c): return [_number(r.get(PRIMARY)) for _,r in rows if _cat(r.get("criterion"))==c and _number(r.get(PRIMARY)) is not None]
def _secondary(rows,name):
    return {c:(lambda v:dict(zip(("finite_count","mean","sample_variance","direction"),(*_sample(v),"maximize"))))([_number(r.get(name)) for _,r in rows if _cat(r.get("criterion"))==c and _number(r.get(name)) is not None]) for c in (REFERENCE,CANDIDATE)}
def certify(path,threshold=None):
    if threshold is not None and (not math.isfinite(threshold) or threshold<0): raise ValueError("threshold must be finite and nonnegative")
    _,rows,bad=_read(path); ref,cand=_group(rows,REFERENCE),_group(rows,CANDIDATE); n1,m1,v1=_sample(ref); n2,m2,v2=_sample(cand); d=None if m1 is None or m2 is None else m2-m1
    out={"comparison":{"reference":REFERENCE,"candidate":CANDIDATE,"outcome":PRIMARY,"direction":"minimize","difference_candidate_minus_reference":d},"reference":{"finite_count":n1,"mean":m1,"sample_variance":v1},"candidate":{"finite_count":n2,"mean":m2,"sample_variance":v2},"secondary":{"ACC+":_secondary(rows,"ACC+"),"PRED40+":_secondary(rows,"PRED40+")},"threshold":threshold,"malformed_rows":bad,"limitations":"Historical configurations are not controlled repeated evaluations; other hyperparameters may differ, so this does not establish causation or guaranteed improvement."}
    if n1<2 or n2<2 or v1 is None or v2 is None or v1/n1+v2/n2==0: out["inference"]={"status":"undefined","reason":"insufficient samples or zero Welch standard error"}; return out
    se=math.sqrt(v1/n1+v2/n2); df=(v1/n1+v2/n2)**2/((v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1)); t=d/se; p=2*(1-t_cdf(abs(t),df)); ci=[d-t_critical(df)*se,d+t_critical(df)*se]
    out["inference"]={"status":"defined","standard_error":se,"degrees_of_freedom":df,"t_statistic":t,"two_sided_p":min(1,max(0,p)),"confidence_interval_95":ci,"method":"Welch t test; Welch-Satterthwaite df; incomplete-beta continued fraction and bisection"}
    out["threshold_interpretation"]=("practical importance undetermined without a threshold" if threshold is None else {"point_estimate_meets":d<=-threshold,"entire_ci_supports":ci[1]<=-threshold,"note":"nonsignificance does not establish equivalence"}); return out
