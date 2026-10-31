import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'source'))
from estimator import *
out=Path(__file__).parent; data=Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv')
rows=valid_rows(load_rows(data)); results=[evaluate(rows,s) for s in range(42,52)]
fields=['seed','train_count','test_count','train_groups','test_groups','model_mae','baseline_mae','difference','negative_predictions']
with open(out/'metrics_detail.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
 for q in results:
  w.writerow({'seed':q['seed'],'train_count':len(q['train']),'test_count':len(q['test']),'train_groups':len(set(map(group_key,q['train']))),'test_groups':len(set(map(group_key,q['test']))),'model_mae':q['model_mae'],'baseline_mae':q['baseline_mae'],'difference':q['difference'],'negative_predictions':sum(p<0 for p in q['pred'])})
  for r,p,a in zip(q['test'],q['pred'],q['actual']): print(q['seed'],r.get('idX'),p,a)
diff=[q['difference'] for q in results]; wins=sum(x < -TIE_TOLERANCE for x in diff); ties=sum(abs(x)<=TIE_TOLERANCE for x in diff); losses=sum(x > TIE_TOLERANCE for x in diff)
print('summary mean',sum(diff)/len(diff),'median',sorted(diff)[len(diff)//2-1:len(diff)//2+1],'wins ties losses',wins,ties,losses)
