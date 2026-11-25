#!/usr/bin/env python3
"""Independent EZR-informed mixed-type effort comparison."""
import csv, math, random, statistics, sys

PREDICTORS = ["prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced"]

def norm(v):
    return "" if v is None else " ".join(str(v).strip().split()).lower()

def load(path):
    out=[]
    with open(path, newline="") as f:
        for pos,row in enumerate(csv.DictReader(f), 2):
            try: effort=float(row["Effort-"])
            except (KeyError,TypeError,ValueError): effort=math.nan
            x={k:norm(row.get(k,"")) for k in PREDICTORS}
            out.append({"csv_position":pos,"idX":row.get("idX",""),"x":x,"y":effort})
    return out

def group_key(row): return tuple(row["x"][k] for k in PREDICTORS)

def split_groups(rows, seed):
    groups={}
    for i,r in enumerate(rows): groups.setdefault(group_key(r),[]).append(i)
    if len(groups)<2: raise ValueError("at least two distinct descriptor groups required")
    keys=list(groups); random.Random(seed).shuffle(keys)
    cut=max(1, min(len(keys)-1, round(len(keys)*.75)))
    train=set(i for k in keys[:cut] for i in groups[k]); test=set(i for k in keys[cut:] for i in groups[k])
    return sorted(train), sorted(test)

class Estimator:
    def fit(self, rows):
        valid=[r for r in rows if math.isfinite(r["y"])]
        if not valid: raise ValueError("no finite training targets")
        self.rows=valid; self.means={}; self.ranges={}
        for k in PREDICTORS:
            vals=[float(r["x"][k]) for r in valid if r["x"][k].replace(".","",1).isdigit()]
            self.means[k]=statistics.mean(vals) if vals else 0.0; self.ranges[k]=(max(vals)-min(vals)) if vals else 1.0
        return self
    def _distance(self,a,b):
        total=0.0
        for k in PREDICTORS:
            av,bv=a["x"][k],b["x"][k]
            try: d=(float(av)-float(bv))/max(self.ranges[k],1e-12)
            except ValueError: d=0.0 if av==bv else 1.0
            total += d*d
        return math.sqrt(total)
    def predict(self,row):
        ranked=sorted((self._distance(row,r),r["y"]) for r in self.rows)
        return sum(y for _,y in ranked[:min(3,len(ranked))])/min(3,len(ranked))

def mae(pred, actual):
    if len(pred)!=len(actual) or not pred: raise ValueError("MAE requires paired nonempty data")
    return sum(abs(a-b) for a,b in zip(pred,actual))/len(pred)

def classify(diff, tolerance=1e-9):
    return "tie" if abs(diff)<=tolerance else ("win" if diff<0 else "loss")

def run(path, seeds=range(42,52)):
    rows=load(path); results=[]; tol=1e-9
    for seed in seeds:
        ti,vi=split_groups(rows,seed); train=[rows[i] for i in ti]; test=[rows[i] for i in vi]
        model=Estimator().fit(train); baseline=statistics.mean(r["y"] for r in train if math.isfinite(r["y"]))
        mp=[model.predict(r) for r in test]; actual=[r["y"] for r in test]
        mm=mae(mp,actual); bm=mae([baseline]*len(test),actual)
        diff=mm-bm; outcome=classify(diff,tol)
        results.append((seed,len(train),len(test),mm,bm,diff,outcome,mp,test,baseline))
    return results

def main(path):
    rs=run(path); print("seed,train_count,test_count,model_mae,baseline_mae,paired_difference,outcome")
    for s,n,m,a,b,d,o,mp,test,_ in rs:
        print(f"{s},{n},{m},{a:.6f},{b:.6f},{d:.6f},{o}")
        for r,p in zip(test,mp): print(f"detail,seed={s},csv_position={r['csv_position']},idX={r['idX']},prediction={p:.6f},observed={r['y']:.6f}")
    ds=[r[5] for r in rs]; print(f"mean_difference={statistics.mean(ds):.6f} median_difference={statistics.median(ds):.6f}")
    print("wins=%d ties=%d losses=%d"%(sum(r[6]=="win" for r in rs),sum(r[6]=="tie" for r in rs),sum(r[6]=="loss" for r in rs)))
    neg=sum(p<0 for r in rs for p in r[7]); print(f"negative_predictions={neg}")
if __name__=="__main__": main(sys.argv[1])
