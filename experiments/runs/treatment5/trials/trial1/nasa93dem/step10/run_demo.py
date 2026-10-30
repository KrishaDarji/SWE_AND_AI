import csv, os, statistics
from source.estimator import load_csv, compare
ROOT=os.path.dirname(os.path.abspath(__file__))
data='/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'
rows=load_csv(data); print('INVALID_TARGETS_EXCLUDED',rows.invalid_count,'TOTAL_ROWS',rows.total_count,'USABLE_ROWS',len(rows)); results=compare(rows)
with open(os.path.join(ROOT,'metrics.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass','runtime_seconds','error_count','notes'])
    # runtime_seconds is patched by the shell runner after /usr/bin/time captures it.
    w.writerow([10,5,1,'software project estimation','race',0,0,1,'RUNTIME_FROM_TIME',0,'10 splits; EZR-inspired mixed categorical distance and seeded grouped splits; repeated splits overlap'])
for r in results:
    print('SEED',r['seed'],'train',len(r['train']),'test',len(r['test']))
    print('positions',[(x['position'],x['idX']) for x in r['test']])
    print('predictions', [round(x,6) for x in r['model_predictions']])
    print('observed',r['observed'],'baseline',round(r['baseline_predictions'][0],6),'model_mae',r['model_mae'],'baseline_mae',r['baseline_mae'],'difference',r['difference'],'negative',r['negative_predictions'])
ds=[r['difference'] for r in results]
print('SUMMARY mean_difference',statistics.mean(ds),'median_difference',statistics.median(ds),'wins',sum(d < -1e-12 for d in ds),'ties',sum(abs(d)<=1e-12 for d in ds),'losses',sum(d > 1e-12 for d in ds))
