#!/usr/bin/env python3
"""Leakage-safe configuration-to-quality prediction for Health-ClosedIssues."""
import csv, math, random, sys
from pathlib import Path

PREDICTORS = ('N_estimators','criterion','Min_sample_leaves','Min_impurity_decrease','Max_depth')
TARGETS = ('MRE-','ACC+')

def _num(x):
    if x is None or str(x).strip() in ('','?'): return None
    try:
        y=float(x); return y if math.isfinite(y) else None
    except (TypeError,ValueError): return None

def normalize_config(c):
    if not isinstance(c, dict) or set(c) != set(PREDICTORS):
        raise ValueError('configuration must contain exactly the five predictor fields')
    out={}
    for k in PREDICTORS:
        if k == 'criterion': out[k] = None if c[k] is None or str(c[k]).strip() in ('','?') else str(c[k]).strip()
        else: out[k] = _num(c[k])
    return tuple(out[k] for k in PREDICTORS)

def _key(row): return normalize_config({k:row.get(k) for k in PREDICTORS})

def _solve(a,b):
    n=len(a); aug=[list(a[i])+[b[i]] for i in range(n)]
    for j in range(n):
        p=max(range(j,n), key=lambda i:abs(aug[i][j]))
        if abs(aug[p][j]) < 1e-12: continue
        aug[j],aug[p]=aug[p],aug[j]; q=aug[j][j]
        aug[j]=[v/q for v in aug[j]]
        for i in range(n):
            if i != j:
                q=aug[i][j]; aug[i]=[aug[i][k]-q*aug[j][k] for k in range(n+1)]
    return [aug[i][-1] for i in range(n)]

class GuessModel:
    def __init__(self, rows, seed=42, test_fraction=.2):
        self.predictor_names=list(PREDICTORS); self.seed=seed
        clean=[]
        for r in rows:
            try: key=_key(r); y=[_num(r.get(t)) for t in TARGETS]
            except ValueError: continue
            if all(v is not None for v in y): clean.append((key,y))
        groups={}
        for key,y in clean: groups.setdefault(key,[]).append(y)
        if len(groups)<2: raise ValueError('insufficient distinct configuration groups')
        keys=list(groups); random.Random(seed).shuffle(keys)
        cut=max(1,min(len(keys)-1,round(len(keys)*(1-test_fraction))))
        self.train_keys=set(keys[:cut]); self.test_keys=set(keys[cut:])
        self.train=[(k,y) for k,y in clean if k in self.train_keys]
        self.test=[(k,y) for k,y in clean if k in self.test_keys]
        if not self.train or not self.test: raise ValueError('empty train or test set')
        self._fit()

    def _fit(self):
        numeric=[i for i,k in enumerate(PREDICTORS) if k != 'criterion']
        vals={i:[k[i] for k,_ in self.train if k[i] is not None] for i in numeric}
        self.mu={i:(sum(v)/len(v) if v else 0.) for i,v in vals.items()}
        self.sd={i:(math.sqrt(sum((x-self.mu[i])**2 for x in v)/max(1,len(v)-1)) or 1.) for i,v in vals.items()}
        self.categories=sorted({k[1] for k,_ in self.train if k[1] is not None})
        def vec(k):
            z=[1.]
            for i in numeric: z.append(((k[i] if k[i] is not None else self.mu[i])-self.mu[i]) / self.sd[i])
            z += [1. if k[1]==c else 0. for c in self.categories]
            return z
        X=[vec(k) for k,_ in self.train]; p=len(X[0]); self.coeffs=[]
        gram=[[sum(x[i]*x[j] for x in X)+(1e-8 if i==j else 0.) for j in range(p)] for i in range(p)]
        for t in range(2): self.coeffs.append(_solve(gram,[sum(x[i]*y[t] for x,(_,y) in zip(X,self.train)) for i in range(p)]))
        self._vec=vec; self.training_means=[sum(y[t] for _,y in self.train)/len(self.train) for t in range(2)]

    def predict(self, config):
        k=normalize_config(config); x=self._vec(k)
        return {t:sum(a*b for a,b in zip(co,x)) for t,co in zip(TARGETS,self.coeffs)}

    def evaluate(self):
        errors=[]
        for t in range(2):
            pred=[self.predict(dict(zip(PREDICTORS,k)))[TARGETS[t]] for k,_ in self.test]
            actual=[y[t] for _,y in self.test]
            errors.append((sum(abs(a-b) for a,b in zip(actual,pred))/len(actual), sum(abs(a-self.training_means[t]) for a in actual)/len(actual)))
        return {TARGETS[i]:{'heldout_mae':errors[i][0],'training_mean_baseline_mae':errors[i][1]} for i in range(2)}

def load_csv(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def main(argv=None):
    argv=argv or sys.argv[1:]; model=GuessModel(load_csv(argv[0] if argv else 'Health-ClosedIssues0000.csv'))
    print({'predictors':model.predictor_names,'train_count':len(model.train),'test_count':len(model.test),'train_groups':len(model.train_keys),'test_groups':len(model.test_keys),'settings':{'seed':model.seed,'test_fraction':.2},'metrics':model.evaluate()})
if __name__=='__main__': main()
