"""Independent EZR-informed categorical nearest-neighbor effort estimator."""
import csv, math, random

PREDICTORS = ['centerX','YearX','prec','flex','resl','team','pmat','rely','data','cplx','ruse','docu','time','stor','pvol','acap','pcap','pcon','apex','plex','ltex','tool','site','sced']
# NASA93 has 24 descriptors; X-suffix metadata are excluded, leaving 22.
PREDICTORS = [x for x in PREDICTORS if x not in ('centerX','YearX')]

def norm(v):
    return '' if v is None else ' '.join(str(v).strip().split()).lower()

class Records(list):
    def __init__(self, *args, invalid_count=0, total_count=0):
        super().__init__(*args); self.invalid_count=invalid_count; self.total_count=total_count

def load_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    out=[]; invalid=0
    for pos, row in enumerate(rows, 2):
        x=tuple(norm(row.get(k,'')) for k in PREDICTORS)
        try: y=float(row['Effort-'])
        except (TypeError, ValueError): y=None
        if y is None or not math.isfinite(y): invalid += 1; continue
        out.append({'position':pos,'idX':row.get('idX',''),'x':x,'y':y})
    return Records(out, invalid_count=invalid, total_count=len(rows))

def groups(rows):
    d={}
    for i,r in enumerate(rows): d.setdefault(r['x'],[]).append(i)
    return d

def split_indices(rows, seed):
    gs=list(groups(rows).values())
    if len(gs)<2: raise ValueError('insufficient descriptor groups')
    rng=random.Random(seed); rng.shuffle(gs)
    target=max(1, round(.75*len(rows))); train=[]; test=[]
    for n,g in enumerate(gs):
        if n < len(gs)-1 and len(train)+len(g) <= target: train.extend(g)
        else: test.extend(g)
    if not train: raise ValueError('empty train set')
    return sorted(train), sorted(test)

def distance(a,b):
    # EZR SYM.dist: equal symbols are zero, unknown symbols are maximally distant.
    return sum(1.0 if (u=='' or v=='' or u!=v) else 0.0 for u,v in zip(a,b)) / len(a)

class Estimator:
    def __init__(self, k=5): self.k=k; self.rows=[]; self.mean=None
    def fit(self, rows):
        valid=[r for r in rows if r['y'] is not None and math.isfinite(r['y'])]
        if not valid: raise ValueError('no valid training targets')
        self.rows=[{'x':r['x'],'y':r['y']} for r in valid]
        self.mean=sum(r['y'] for r in valid)/len(valid); return self
    def predict_one(self,x):
        if not self.rows: raise ValueError('unfitted estimator')
        near=sorted((distance(x,r['x']),r['y']) for r in self.rows)[:self.k]
        if near[0][0]==0: return sum(y for d,y in near if d==0)/sum(1 for d,y in near if d==0)
        weights=[1/(d+1e-12) for d,y in near]
        return sum(w*y for w,(d,y) in zip(weights,near))/sum(weights)
    def predict(self, rows): return [self.predict_one(r['x']) for r in rows]

def mae(pred, actual):
    if len(pred)!=len(actual) or not actual: raise ValueError('invalid MAE inputs')
    return sum(abs(a-b) for a,b in zip(pred,actual))/len(actual)

def compare(rows, seeds=range(42,52), tolerance=1e-12):
    rows=[r for r in rows if r.get('y') is not None and math.isfinite(r['y'])]
    if not rows: raise ValueError('no usable finite effort targets')
    result=[]
    for seed in seeds:
        ti,vi=split_indices(rows,seed); train=[rows[i] for i in ti]; test=[rows[i] for i in vi]
        model=Estimator().fit(train); mp=model.predict(test); base=sum(r['y'] for r in train)/len(train)
        actual=[r['y'] for r in test]
        mm, bm=mae(mp,actual), mae([base]*len(test),actual)
        result.append({'seed':seed,'train':train,'test':test,'model_predictions':mp,'observed':actual,'baseline_predictions':[base]*len(test),'model_mae':mm,'baseline_mae':bm,'difference':mm-bm,'negative_predictions':sum(p<0 for p in mp),'tied':abs(mm-bm)<=tolerance})
    return result
