import csv, math, random, statistics

DESCRIPTORS = ['prec','flex','resl','team','pmat','rely','data','cplx','ruse','docu','time','stor','pvol','acap','pcap','pcon','apex','plex','ltex','tool','site','sced']
TARGET = 'Effort-'

def norm(x):
    return ' '.join(str(x).strip().lower().split())

def load_rows(path):
    out=[]
    with open(path, newline='') as f:
        for original, raw in enumerate(csv.DictReader(f), 2):
            try: y=float(raw[TARGET].strip())
            except (ValueError, TypeError): continue
            if not math.isfinite(y): continue
            out.append({'line':original, 'id':raw['idX'].strip(), 'x':tuple(norm(raw[k]) for k in DESCRIPTORS), 'y':y})
    return out

def split_groups(rows, seed=42):
    groups={}
    for r in rows: groups.setdefault(r['x'], []).append(r)
    keys=list(groups); random.Random(seed).shuffle(keys)
    target=round(len(rows)*.75); train=[]; test=[]
    for k in keys:
        bucket=groups[k]
        if len(train)<target: train.extend(bucket)
        else: test.extend(bucket)
    if not test: raise ValueError('insufficient descriptor groups for a held-out set')
    return train, test

class Estimator:
    def __init__(self, train):
        if len(train)<2: raise ValueError('insufficient training rows')
        self.train=train; self.reference=statistics.median(r['y'] for r in train)
        self.values=[set(r['x'][i] for r in train) for i in range(len(DESCRIPTORS))]
    def predict(self, proposal):
        if not isinstance(proposal, dict): raise ValueError('proposal must be a mapping')
        missing=[k for k in DESCRIPTORS if k not in proposal or not norm(proposal[k])]
        if missing: raise ValueError('invalid proposal; missing descriptors: '+','.join(missing))
        x=tuple(norm(proposal[k]) for k in DESCRIPTORS)
        d=[]
        for r in self.train:
            dist=sum(a!=b for a,b in zip(x,r['x']))
            d.append((dist, r['id'], r))
        d.sort(key=lambda z:(z[0],z[1]))
        best=d[0][0]; peers=[z[2] for z in d if z[0]==best]
        estimate=statistics.median(r['y'] for r in peers)
        evidence='model_behavior: median effort of %d nearest training row(s) at descriptor distance %d' % (len(peers),best)
        unseen=[DESCRIPTORS[i] for i,v in enumerate(x) if v not in self.values[i]]
        return {'estimate':estimate,'reference':self.reference,'evidence':evidence,'unseen_descriptors':unseen,'nearest_ids':[r['id'] for r in peers]}
    def association(self, descriptor, value):
        if descriptor not in DESCRIPTORS: raise ValueError('unknown descriptor')
        i=DESCRIPTORS.index(descriptor); v=norm(value)
        a=[r['y'] for r in self.train if r['x'][i]==v]
        b=[r['y'] for r in self.train if r['x'][i]!=v]
        return {'label':'historical_association','descriptor':descriptor,'value':v,'matching_n':len(a),'matching_median':statistics.median(a) if a else None,'other_median':statistics.median(b) if b else None}

def demonstrate(dataset):
    rows=load_rows(dataset); train,test=split_groups(rows); model=Estimator(train); chosen=min(test,key=lambda r:(r['line'],r['id']))
    proposal=dict(zip(DESCRIPTORS,chosen['x'])); frozen=model.predict(proposal)
    audit={'label':'measured_audit_value','record_id':chosen['id'],'observed_effort':chosen['y']}
    print('TRAIN_ROWS',len(train)); print('TEST_ROWS',len(test)); print('HELD_OUT_RECORD',chosen['id']); print('PROPOSAL_DESCRIPTORS',len(proposal)); print('FROZEN_ESTIMATE',frozen['estimate']); print('TRAINING_MEDIAN_REFERENCE',frozen['reference']); print('EVIDENCE',frozen['evidence']); print('UNSEEN_DESCRIPTORS',','.join(frozen['unseen_descriptors']) or 'none'); print('AUDIT',audit)

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2: raise SystemExit('usage: python estimator.py DATASET')
    demonstrate(sys.argv[1])
