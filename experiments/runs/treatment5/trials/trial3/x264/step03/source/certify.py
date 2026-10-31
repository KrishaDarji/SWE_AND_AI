#!/usr/bin/env python3
"""Independent Welch comparison for two fixed configuration groups."""
import argparse, csv, math, sys

def _betacf(a, b, x):
    qab, qap, qam = a+b, a+1, a-1
    c, d = 1.0, 1.0-qab*x/qap
    d = max(abs(d), 1e-300) * (1 if d >= 0 else -1); d = 1/d
    h = d
    for m in range(1, 301):
        m2=2*m
        aa=m*(b-m)*x/((qam+m2)*(a+m2))
        d=1+aa*d; d=max(abs(d),1e-300)*(1 if d>=0 else -1); c=1+aa/c; c=max(abs(c),1e-300)*(1 if c>=0 else -1); d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=max(abs(d),1e-300)*(1 if d>=0 else -1); c=1+aa/c; c=max(abs(c),1e-300)*(1 if c>=0 else -1); d=1/d
        delta=d*c; h*=delta
        if abs(delta-1)<3e-14: break
    return h

def ibeta(a,b,x):
    if not (0 <= x <= 1): raise ValueError("beta argument out of range")
    if x in (0,1): return x
    front=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return front*_betacf(a,b,x)/a
    return 1-front*_betacf(b,a,1-x)/b

def t_cdf(t, df):
    if df <= 0: raise ValueError("degrees of freedom must be positive")
    if t == 0: return .5
    z=df/(df+t*t); tail=.5*ibeta(df/2,.5,z)
    return 1-tail if t > 0 else tail

def t_quantile(p, df):
    if not 0 < p < 1: raise ValueError("probability out of range")
    lo,hi=-1.,1.
    while t_cdf(lo,df)>p: lo*=2
    while t_cdf(hi,df)<p: hi*=2
    for _ in range(120):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2

def summarize(values):
    if not values: raise ValueError("group has no finite observations")
    n=len(values); mean=math.fsum(values)/n
    var=math.fsum((x-mean)**2 for x in values)/(n-1) if n>1 else float('nan')
    return n,mean,var

def compare(a,b,threshold=5.0):
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<0: raise ValueError("threshold must be finite and nonnegative")
    na,ma,va=summarize(a); nb,mb,vb=summarize(b)
    if na<2 or nb<2: raise ValueError("each group needs at least two finite observations")
    diff=ma-mb; se2=va/na+vb/nb
    if not math.isfinite(se2) or se2<=0: raise ValueError("Welch standard error is zero or invalid")
    se=math.sqrt(se2); df=se2**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1)); t=diff/se
    p=2*min(t_cdf(t,df),1-t_cdf(t,df)); crit=t_quantile(.975,df); lo=diff-crit*se; hi=diff+crit*se
    return dict(n_a=na,n_b=nb,mean_a=ma,mean_b=mb,difference_a_minus_b=diff,df=df,t=t,p_two_sided=p,ci95_low=lo,ci95_high=hi,threshold=threshold,significant_zero=p<.05,point_exceeds_threshold=abs(diff)>threshold,interval_beyond_threshold=(lo>threshold or hi< -threshold),practical_interpretation="meaningfully faster" if hi < -threshold else ("meaningfully slower" if lo > threshold else "practical importance uncertain"))

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--threshold",type=float,default=5.0); args=ap.parse_args(argv)
    a=[]; b=[]; skipped=0
    with open(args.csv,newline="") as f:
        for row in csv.DictReader(f):
            try: value=float(row["Performance-"])
            except (KeyError,TypeError,ValueError): skipped+=1; continue
            if not math.isfinite(value): skipped+=1; continue
            if row.get("no_8x8dct")=="0": a.append(value)
            elif row.get("no_8x8dct")=="1": b.append(value)
    result=compare(a,b,args.threshold); result["skipped_nonfinite_or_invalid"]=skipped; result["better_group"]="A" if result["difference_a_minus_b"]<0 else ("B" if result["difference_a_minus_b"]>0 else "tie")
    print("Welch comparison: A=no_8x8dct=0, B=no_8x8dct=1; lower Performance- is better")
    for k,v in result.items(): print(f"{k}: {v}")
    return 0
if __name__ == "__main__": sys.exit(main())
