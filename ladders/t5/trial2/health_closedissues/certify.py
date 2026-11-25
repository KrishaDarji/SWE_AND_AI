#!/usr/bin/env python3
"""Certify a fixed squared_error versus absolute_error comparison."""
import argparse, csv, math, json

def mean_var(xs):
    n=0; m=0.0; s=0.0
    for x in xs:
        n += 1; d=x-m; m += d/n; s += d*(x-m)
    return n,m,(s/(n-1) if n>1 else 0.0)

def betacf(a,b,x):
    qab=a+b; qap=a+1; qam=a-1; c=1.0; d=1-qab*x/qap
    d=(1 if d>=0 else -1)*max(abs(d),1e-300); h=1/d
    for m in range(1,10001):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=(1 if d>=0 else -1)*max(abs(d),1e-300); c=1+aa/c; c=(1 if c>=0 else -1)*max(abs(c),1e-300); d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=(1 if d>=0 else -1)*max(abs(d),1e-300); c=1+aa/c; c=(1 if c>=0 else -1)*max(abs(c),1e-300); d=1/d; de=d*c; h*=de
        if abs(de-1)<3e-14: break
    return h

def ibeta(x,a,b):
    if x<=0: return 0.0
    if x>=1: return 1.0
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return bt*betacf(a,b,x)/a
    return 1-bt*betacf(b,a,1-x)/b

def t_cdf(t,df):
    if not math.isfinite(t) or df<=0: return float('nan')
    if df>1000: return .5*math.erfc(-t/math.sqrt(2))
    x=df/(df+t*t); p=.5*ibeta(x,df/2,.5)
    return 1-p if t>=0 else p

def t_quantile(prob,df):
    if not (0<prob<1 and df>0): return float('nan')
    lo,hi=-1000.0,1000.0
    for _ in range(160):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<prob: lo=mid
        else: hi=mid
    return (lo+hi)/2

def welch(a,b,threshold=None):
    na,ma,va=mean_var(a); nb,mb,vb=mean_var(b); out={'n_reference':na,'n_candidate':nb,'mean_reference':ma,'mean_candidate':mb,'difference_candidate_minus_reference':mb-ma}
    if na<2 or nb<2 or (va==0 and vb==0): out.update(status='undefined',reason='insufficient groups or both variances zero'); return out
    se2=va/na+vb/nb
    if se2<=0: out.update(status='undefined',reason='zero standard error'); return out
    df=se2*se2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    se=math.sqrt(se2); t=(mb-ma)/se; crit=t_quantile(.975,df); p=(math.erfc(abs(t)/math.sqrt(2)) if df>1000 else 2*min(t_cdf(t,df),1-t_cdf(t,df)))
    out.update(status='ok',variance_reference=va,variance_candidate=vb,standard_error=se,degrees_freedom=df,t=t,p_two_sided=max(0,min(1,p)),ci95_low=mb-ma-crit*se,ci95_high=mb-ma+crit*se)
    if threshold is not None:
        out['threshold']=threshold; out['point_improvement_beyond_threshold']=(ma-mb)>threshold; out['entire_interval_favorable_beyond_threshold']=out['ci95_high'] < -threshold
    return out

def read(path):
    groups={}; secondary={}
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            c=r.get('criterion',''); raw=r.get('MRE-','')
            try: x=float(raw)
            except (TypeError,ValueError): continue
            if not math.isfinite(x): continue
            groups.setdefault(c,[]).append(x)
            secondary.setdefault(c,[]).append((float(r['ACC+']),float(r['PRED40+'])))
    return groups,secondary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv_path'); ap.add_argument('--threshold',type=float); args=ap.parse_args()
    if args.threshold is not None and (not math.isfinite(args.threshold) or args.threshold<0): ap.error('threshold must be finite and nonnegative')
    g,s=read(args.csv_path); ref=g.get('squared_error',[]); cand=g.get('absolute_error',[]); result={'primary':'MRE-; lower is better','reference_definition':'criterion=squared_error','candidate_definition':'criterion=absolute_error','comparison':welch(ref,cand,args.threshold),'secondary':{}}
    for k in ('squared_error','absolute_error'):
        z=s.get(k,[]); result['secondary'][k]={'count':len(z),'mean_ACC+':sum(x[0] for x in z)/len(z) if z else None,'mean_PRED40+':sum(x[1] for x in z)/len(z) if z else None}
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__': main()
