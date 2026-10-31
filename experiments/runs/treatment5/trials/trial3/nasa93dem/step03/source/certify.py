#!/usr/bin/env python3
"""Welch comparison of observed Effort- for tool=n and tool=h."""
import csv, math, argparse

def _beta_cf(a, b, x):
    qab, qap, qam = a+b, a+1.0, a-1.0
    c, d, h = 1.0, 1.0-qab*x/qap, 1.0
    if abs(d) < 3e-300: d=3e-300
    d=1.0/d; h=d
    for m in range(1, 10000):
        m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d
        if abs(d)<3e-300:d=3e-300
        c=1+aa/c
        if abs(c)<3e-300:c=3e-300
        d=1/d; h*=d*c; aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d
        if abs(d)<3e-300:d=3e-300
        c=1+aa/c
        if abs(c)<3e-300:c=3e-300
        d=1/d; delta=d*c; h*=delta
        if abs(delta-1)<3e-14: break
    return h

def _ibeta(a,b,x):
    if x<=0:return 0.0
    if x>=1:return 1.0
    front=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return front*_beta_cf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-front*_beta_cf(b,a,1-x)/b

def t_cdf(t, df):
    x=df/(df+t*t); tail=0.5*_ibeta(df/2,0.5,x)
    return 1-tail if t>=0 else tail

def normal_quantile(p):
    # Acklam rational approximation; max error about 4.5e-4 in this use.
    a=[-39.6968302866538,220.946098424521,-275.928510446969,138.357751867269,-30.6647980661472,2.50662827745924]
    b=[-54.4760987982241,161.585836858041,-155.698979859887,66.8013118877197,-13.2806815528857]
    c=[-.00778489400243029,-.322396458041136,-2.40075827716184,-2.54973253934373,4.37466414146497,2.93816398269878]
    d=[.00778469570904146,.32246712907004,2.445134137143,3.75440866190742]
    if p<.02425:
        q=math.sqrt(-2*math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p>1-.02425:return -normal_quantile(1-p)
    q=p-.5;r=q*q; num=((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q); den=(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1); return num/den

def mean_var(xs):
    n=0; mean=0.; m2=0.
    for x in xs:
        n+=1; d=x-mean; mean+=d/n; m2+=d*(x-mean)
    return n,mean,(m2/(n-1) if n>1 else 0.)

def compare(a,b,threshold):
    if not math.isfinite(threshold) or threshold<0: raise ValueError("threshold must be finite and nonnegative")
    a=[x for x in a if math.isfinite(x)]; b=[x for x in b if math.isfinite(x)]
    if len(a)<2 or len(b)<2: raise ValueError("each group needs at least two finite outcomes")
    na,ma,va=mean_var(a); nb,mb,vb=mean_var(b); diff=ma-mb
    se2=va/na+vb/nb
    if se2==0: return dict(n_a=na,n_b=nb,mean_a=ma,mean_b=mb,difference=diff,df=float('inf'),se=0,ci_low=diff,ci_high=diff,p_value=(0.0 if diff else 1.0),threshold=threshold,statistical_difference=bool(diff),magnitude_exceeds=abs(diff)>threshold,entirely_practical=abs(diff)>threshold)
    df=se2*se2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1)); se=math.sqrt(se2); t=diff/se; crit=normal_quantile(.975)
    # Student-t critical value via monotone bisection of the CDF.
    lo,hi=0.,10.;
    for _ in range(70):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<.975:lo=mid
        else:hi=mid
    crit=(lo+hi)/2; p=2*(1-t_cdf(abs(t),df))
    return dict(n_a=na,n_b=nb,mean_a=ma,mean_b=mb,difference=diff,df=df,se=se,ci_low=diff-crit*se,ci_high=diff+crit*se,p_value=p,threshold=threshold,statistical_difference=p<.05,magnitude_exceeds=abs(diff)>threshold,entirely_practical=(diff>threshold or diff< -threshold) and not (diff-crit*se<=threshold and diff+crit*se>=-threshold))

def load_groups(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        rows=csv.DictReader(f, skipinitialspace=True); a=[];b=[]
        for row in rows:
            tool=row['tool'].strip().lower(); raw=row['Effort-'].strip()
            try:x=float(raw)
            except (ValueError,TypeError):continue
            if tool=='n':a.append(x)
            elif tool=='h':b.append(x)
    return a,b

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--threshold',type=float,required=True); args=ap.parse_args()
    a,b=load_groups(args.dataset); r=compare(a,b,args.threshold)
    print('A: tool=n (observed Effort-); B: tool=h (observed Effort-)')
    for k,v in r.items(): print(f'{k}={v}')
    print('Interpretation: positive A-minus-B favors B; negative favors A. Effort- is not verified monetary cost.')
    print('Illustrative threshold is 10% of tool=n mean; scope, skew, imbalance, and non-causal observational limits apply.')
if __name__=='__main__':main()
