#!/usr/bin/env python3
"""Independent certification comparison for the Health closed-issues data."""
import csv, math, argparse

def finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False

def mean(xs): return sum(xs)/len(xs)
def variance(xs):
    m=mean(xs); return sum((x-m)**2 for x in xs)/(len(xs)-1)

def betacf(a,b,x):
    qab=a+b; qap=a+1; qam=a-1; c=1.; d=1-qab*x/qap
    d=max(abs(d),1e-300) * (1 if d >= 0 else -1); d=1/d; h=d
    for m in range(1,10001):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=1e-300 if abs(d)<1e-300 else d; c=1+aa/c; c=1e-300 if abs(c)<1e-300 else c; d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=1e-300 if abs(d)<1e-300 else d; c=1+aa/c; c=1e-300 if abs(c)<1e-300 else c; d=1/d; delta=d*c; h*=delta
        if abs(delta-1)<3e-14: break
    return h

def ibeta(a,b,x):
    if x<=0: return 0.
    if x>=1: return 1.
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*betacf(b,a,1-x)/b

def student_cdf(t,df):
    z=df/(df+t*t); p=.5*ibeta(df/2,.5,z)
    return 1-p if t>=0 else p

def student_quantile(p,df):
    lo,hi=-100.,100.
    for _ in range(160):
        mid=(lo+hi)/2
        if student_cdf(mid,df)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2

def welch(a,b,alpha=.05):
    if len(a)<2 or len(b)<2: return None
    va,vb=variance(a),variance(b); se2=va/len(a)+vb/len(b)
    if se2==0: return None
    df=se2*se2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1)) if va or vb else None
    if not df or not math.isfinite(df): return None
    d=mean(b)-mean(a); se=math.sqrt(se2); crit=student_quantile(1-alpha/2,df)
    t=d/se; p=2*(1-student_cdf(abs(t),df))
    return {'difference':d,'df':df,'se':se,'t':t,'p':max(0.,min(1.,p)),'ci_low':d-crit*se,'ci_high':d+crit*se}

def load(path):
    groups={'squared_error':[],'absolute_error':[]}; secondary={k:[] for k in ('ACC+','PRED40+')}
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            c=row.get('criterion','')
            if c in groups and finite(row.get('MRE-')): groups[c].append(float(row['MRE-']))
            if c in groups:
                for k in secondary:
                    if finite(row.get(k)): secondary[k].append((c,float(row[k])))
    return groups,secondary

def summarize(path,threshold=None):
    g,s=load(path); ref,can=g['squared_error'],g['absolute_error']; result={'definitions':'MRE- is primary; candidate-minus-reference, lower is better. ACC+ and PRED40+ are descriptive only.','counts':{'reference':len(ref),'candidate':len(can)},'means':{'reference':mean(ref) if ref else None,'candidate':mean(can) if can else None}}
    result['welch']=welch(ref,can)
    if threshold is not None and (not finite(threshold) or threshold<0): raise ValueError('threshold must be finite and nonnegative')
    result['practical_threshold']=threshold
    if threshold is None: result['practical_importance']='undetermined (no threshold supplied)'
    elif result['welch']:
        d=result['welch']['difference']; result['practical_importance']={'point_beyond_threshold':d < -threshold,'interval_entirely_beyond_threshold':result['welch']['ci_high'] < -threshold}
    result['secondary']={}
    for k,vals in s.items():
        result['secondary'][k]={c:{'count':len([x for q,x in vals if q==c]),'mean':mean([x for q,x in vals if q==c]) if [x for q,x in vals if q==c] else None} for c in ('squared_error','absolute_error')}
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--threshold',type=float); args=ap.parse_args()
    print('Health ClosedIssues criterion certification'); print('Assumptions: independent configuration observations; finite MRE- values only; Welch unequal-variance inference. Other settings may confound differences; no causal, repeated-run-noise, equivalence, or guaranteed-gain claim.')
    print(result_to_text(summarize(args.dataset,args.threshold)))

def result_to_text(r):
    return repr(r)
if __name__=='__main__': main()
