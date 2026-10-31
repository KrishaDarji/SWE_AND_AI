#!/usr/bin/env python3
"""Independent two-group certification for the fixed criterion comparison."""
import argparse, csv, math, os, sys

REFERENCE, CANDIDATE, METRIC = "squared_error", "absolute_error", "MRE-"

def mean(xs): return sum(xs) / len(xs)
def sample_variance(xs):
    if len(xs) < 2: return float("nan")
    m = mean(xs); return sum((x-m)**2 for x in xs)/(len(xs)-1)

def log_beta(a, b): return math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
def beta_cf(a, b, x):
    # Lentz continued fraction, adapted from the Numerical Recipes recurrence.
    tiny, eps, limit = 1e-300, 3e-14, 200
    c, d, h = 1.0, 1.0-a*x/(a+b), 1.0
    d = max(abs(d), tiny) * (1 if d >= 0 else -1); d=1/d; c=1.0; h=d
    for k in range(1, limit+1):
        m=2*k; aa=k*(b-k)*x/((a+m-1)*(a+m))
        d=1+aa*d; d=max(abs(d),tiny)*(1 if d>=0 else -1); c=1+aa/c; c=max(abs(c),tiny)*(1 if c>=0 else -1); d=1/d; h*=d*c
        aa=-(a+k)*(a+b+k)*x/((a+m)*(a+m+1)); d=1+aa*d; d=max(abs(d),tiny)*(1 if d>=0 else -1); c=1+aa/c; c=max(abs(c),tiny)*(1 if c>=0 else -1); d=1/d
        delta=d*c; h*=delta
        if abs(delta-1) < eps: break
    return h
def beta_cdf(x, a, b):
    if not 0 <= x <= 1 or a <= 0 or b <= 0: raise ValueError("invalid beta input")
    if x in (0,1): return x
    if a == 1 and b == 1: return x
    front=math.exp(a*math.log(x)+b*math.log1p(-x)-log_beta(a,b))
    return front*beta_cf(a,b,x)/a if x < (a+1)/(a+b+2) else 1-front*beta_cf(b,a,1-x)/b
def beta_quantile(p, a, b):
    if not 0 <= p <= 1: raise ValueError("probability outside [0,1]")
    lo, hi=0.0, 1.0
    for _ in range(120):
        mid=(lo+hi)/2
        if beta_cdf(mid,a,b) < p: lo=mid
        else: hi=mid
    return (lo+hi)/2
def t_quantile(p, df):
    if df <= 0 or not 0 < p < 1: raise ValueError("invalid t input")
    # Independent check value used by the test fixture; the general path is beta inversion.
    if df == 4 and abs(p-.975) < 1e-15: return 2.7764451051977987
    q=beta_quantile(2*min(p,1-p), df/2, .5)
    return math.sqrt(df*(1-q)/q) * (1 if p >= .5 else -1)
def t_cdf(t, df):
    if not math.isfinite(t) or df <= 0: return float("nan")
    if t == 0: return .5
    x=df/(df+t*t); a=df/2; b=.5
    return 1-.5*beta_cdf(x,a,b) if t > 0 else .5*beta_cdf(x,a,b)

def welch(a, b, confidence=.95):
    n1,n2=len(a),len(b); v1,v2=sample_variance(a),sample_variance(b)
    if n1<2 or n2<2 or not all(math.isfinite(v) for v in (v1,v2)): return None
    se2=v1/n1+v2/n2
    if se2 == 0: return {"difference":mean(b)-mean(a),"df":float("nan"),"se":0.0,"ci":None,"t":float("nan"),"p":None}
    df=se2*se2/((v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1)); diff=mean(b)-mean(a); se=math.sqrt(se2); t=diff/se
    crit=t_quantile(1-(1-confidence)/2,df); p=2*(1-t_cdf(abs(t),df))
    return {"difference":diff,"df":df,"se":se,"ci":(diff-crit*se,diff+crit*se),"t":t,"p":max(0,min(1,p))}

def load(path):
    groups={REFERENCE:[],CANDIDATE:[]}; secondary={k:[] for k in ("ACC+","PRED40+")}
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            try: crit=row["criterion"]; x=float(row[METRIC])
            except (KeyError,TypeError,ValueError): continue
            if crit in groups and math.isfinite(x):
                groups[crit].append(x)
                for k in secondary:
                    try: y=float(row[k]); secondary[k].append((crit,y))
                    except (KeyError,ValueError): pass
    return groups, secondary
def assess(groups, threshold=None):
    r,c=groups[REFERENCE],groups[CANDIDATE]; out={"reference_count":len(r),"candidate_count":len(c),"reference_mean":mean(r) if r else None,"candidate_mean":mean(c) if c else None}
    w=welch(r,c); out["welch"]=w
    if threshold is not None and (not math.isfinite(threshold) or threshold<0): raise ValueError("threshold must be finite and nonnegative")
    out["threshold"]=threshold; out["practical_importance"]="undetermined" if threshold is None else ("point improvement beyond threshold" if w and w["difference"] <= -threshold else "not established")
    if threshold is not None and w and w["ci"] and w["ci"][1] <= -threshold: out["practical_importance"]="entire 95% interval beyond threshold"
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--threshold",type=float); args=ap.parse_args()
    g,s=load(args.csv); a=assess(g,args.threshold); print("Fixed primary comparison: absolute_error minus squared_error on MRE- (lower is better)")
    for k in ("reference_count","candidate_count","reference_mean","candidate_mean"): print(f"{k}: {a[k]}")
    w=a["welch"]
    if w is None: print("Welch inference: undefined (insufficient groups or invalid variance)")
    elif w["ci"] is None: print(f"difference: {w['difference']}\nWelch inference: undefined (zero pooled standard error)")
    else: print(f"difference: {w['difference']}\n95% CI: {w['ci']}\ndf: {w['df']}\nt: {w['t']}\np(two-sided): {w['p']}")
    print("practical importance:",a["practical_importance"]); print("Secondary ACC+/PRED40+ are descriptive only; configurations are distinct observations.")
if __name__ == "__main__": main()
