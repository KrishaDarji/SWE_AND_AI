#!/usr/bin/env python3
"""Independent standard-library Welch certification for two configuration groups."""
import csv, math, sys

EPS = 1e-14

def _beta_cf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = max(abs(d), 1e-300) * (1 if d >= 0 else -1); d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = (1e-300 if abs(d) < 1e-300 else d)
        c = 1.0 + aa / c; c = (1e-300 if abs(c) < 1e-300 else c)
        d, c = 1.0 / d, c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = (1e-300 if abs(d) < 1e-300 else d)
        c = 1.0 + aa / c; c = (1e-300 if abs(c) < 1e-300 else c)
        d, c = 1.0 / d, c
        delta = d * c; h *= delta
        if abs(delta - 1.0) <= 3e-14: break
    return h

def regularized_beta(x, a, b):
    if not (0 <= x <= 1) or min(a, b) <= 0: raise ValueError("invalid beta arguments")
    if x == 0: return 0.0
    if x == 1: return 1.0
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return front * _beta_cf(a,b,x) / a
    return 1.0 - front * _beta_cf(b,a,1-x) / b

def t_cdf(t, df):
    if not (df > 0 and math.isfinite(t)): raise ValueError("invalid t arguments")
    if t == 0: return 0.5
    x = df / (df + t*t); tail = 0.5 * regularized_beta(x, df/2, 0.5)
    return 1-tail if t > 0 else tail

def inverse_t(p, df):
    if not (0 < p < 1 and df > 0): raise ValueError("invalid quantile arguments")
    lo, hi = -1.0, 1.0
    while t_cdf(lo,df) > p: lo *= 2
    while t_cdf(hi,df) < p: hi *= 2
    for _ in range(100):
        mid=(lo+hi)/2
        if t_cdf(mid,df) < p: lo=mid
        else: hi=mid
    return (lo+hi)/2

def summarize(values):
    xs=[float(v) for v in values if math.isfinite(float(v))]
    if not xs: raise ValueError("group has no finite observations")
    mean=math.fsum(xs)/len(xs)
    var=math.fsum((x-mean)**2 for x in xs)/(len(xs)-1) if len(xs)>1 else 0.0
    return len(xs), mean, var

def welch(a,b,threshold=5.0):
    if not (math.isfinite(threshold) and threshold >= 0): raise ValueError("threshold must be finite and nonnegative")
    na,ma,va=summarize(a); nb,mb,vb=summarize(b)
    if na < 2 or nb < 2: raise ValueError("each group needs at least two finite observations")
    se2=va/na+vb/nb
    if se2 == 0: raise ValueError("zero standard error")
    diff=ma-mb; se=math.sqrt(se2); df=se2*se2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    stat=diff/se; p=2*min(t_cdf(stat,df),1-t_cdf(stat,df)); crit=inverse_t(.975,df); half=crit*se
    lo,hi=diff-half,diff+half
    return dict(n_a=na,n_b=nb,mean_a=ma,mean_b=mb,variance_a=va,variance_b=vb,difference=diff,df=df,t=stat,p_value=p,ci_low=lo,ci_high=hi,threshold=threshold,zero_difference_evidence=(p<.05),point_exceeds_threshold=abs(diff)>threshold,interval_beyond_threshold=(lo>threshold or hi < -threshold),practical_uncertain=(lo <= threshold and hi >= -threshold))

def load_groups(path):
    a,b=[],[]
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            try: value=float(row['Performance-'])
            except (KeyError,TypeError,ValueError): continue
            if not math.isfinite(value): continue
            (a if row.get('no_8x8dct') == '0' else b if row.get('no_8x8dct') == '1' else []).append(value)
    if not a or not b: raise ValueError('both comparison groups require finite values')
    return a,b

def main(path):
    a,b=load_groups(path); r=welch(a,b)
    print('A: no_8x8dct=0 (faster candidate); B: no_8x8dct=1')
    for k in ('n_a','n_b','mean_a','mean_b','difference','t','df','p_value','ci_low','ci_high','threshold','zero_difference_evidence','point_exceeds_threshold','interval_beyond_threshold','practical_uncertain'): print(f'{k}={r[k]}')
    print('Lower Performance- is better; rows are configurations, not repeated-run measurements.')
    print('Statistical difference and practical importance are reported separately; crossing the threshold makes practical importance uncertain.')

if __name__ == '__main__':
    if len(sys.argv) != 2: raise SystemExit('usage: certify.py DATASET')
    main(sys.argv[1])
