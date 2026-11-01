"""Welch certification for two fixed X264 configurations."""
import csv, math, json, sys
try:
    from .x264_summary import _number
except ImportError:
    from x264_summary import _number

TAU_DEFAULT = 5.0

def _clean(path, column, level):
    with open(path, newline="", encoding="utf-8-sig") as h:
        rows = csv.DictReader(h)
        if not rows.fieldnames or column not in rows.fieldnames or "Performance-" not in rows.fieldnames:
            raise ValueError("dataset must contain no_8x8dct and Performance-")
        out=[]
        for row in rows:
            x=_number(row.get(column, "")); y=_number(row.get("Performance-", ""))
            if x is not None and y is not None and x == level: out.append(y)
        return out

def _mean_var(values):
    if not values: raise ValueError("each group needs at least one finite observation")
    mean=0.0; m2=0.0
    for n, value in enumerate(values, 1):
        delta=value-mean; mean += delta/n; m2 += delta*(value-mean)
    return mean, (m2/(len(values)-1) if len(values)>1 else 0.0)

def _betacf(a,b,x):
    qab=a+b; qap=a+1.0; qam=a-1.0; c=1.0; d=1.0-qab*x/qap
    if abs(d)<3e-30: d=3e-30
    d=1.0/d; h=d
    for m in range(1,10000):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=3e-30 if abs(d)<3e-30 else d; c=1+aa/c; c=3e-30 if abs(c)<3e-30 else c; d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=3e-30 if abs(d)<3e-30 else d; c=1+aa/c; c=3e-30 if abs(c)<3e-30 else c; d=1/d; delta=d*c; h*=delta
        if abs(delta-1)<3e-14: break
    return h

def _betai(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    front=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return front*_betacf(a,b,x)/a if x < (a+1)/(a+b+2) else 1-front*_betacf(b,a,1-x)/b

def _t_cdf(t, df):
    x=df/(df+t*t); tail=0.5*_betai(df/2, 0.5, x)
    return 1-tail if t>=0 else tail

def _t_quantile(p, df):
    if not 0<p<1 or df<=0: raise ValueError("invalid probability or degrees of freedom")
    lo,hi=-100.0,100.0
    for _ in range(120):
        mid=(lo+hi)/2
        if _t_cdf(mid,df)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2

def certify(path, threshold=TAU_DEFAULT):
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<0: raise ValueError("threshold must be nonnegative and finite")
    a=_clean(path,"no_8x8dct",0); b=_clean(path,"no_8x8dct",1)
    if len(a)<2 or len(b)<2: raise ValueError("each group needs at least two finite observations")
    ma,va=_mean_var(a); mb,vb=_mean_var(b); diff=ma-mb
    se2=va/len(a)+vb/len(b)
    if se2==0: raise ValueError("zero standard error: Welch test is undefined")
    df=se2*se2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    se=math.sqrt(se2); crit=_t_quantile(.975,df); lo=diff-crit*se; hi=diff+crit*se
    p=2*min(_t_cdf(diff/se,df),1-_t_cdf(diff/se,df))
    return {"A":{"definition":"no_8x8dct=0","n":len(a),"mean":ma},"B":{"definition":"no_8x8dct=1","n":len(b),"mean":mb},"mean_difference_A_minus_B":diff,"standard_error":se,"degrees_of_freedom":df,"t_statistic":diff/se,"p_value_two_sided":p,"confidence_interval_95":[lo,hi],"threshold":threshold,"point_estimate_exceeds_threshold":abs(diff)>threshold,"interval_entirely_beyond_threshold":lo>threshold or hi < -threshold,"interval_crosses_practical_threshold":lo<=threshold and hi>=-threshold,"interpretation":"lower Performance- is better; rows are configurations, not repeated-run measurements"}

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: x264_certify.py DATASET")
    print(json.dumps(certify(sys.argv[1]), indent=2, sort_keys=True))
