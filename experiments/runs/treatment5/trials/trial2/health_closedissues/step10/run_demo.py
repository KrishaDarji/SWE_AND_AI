import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))
import selector

DATASET = '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv'
OUT = os.path.dirname(__file__)
rows = selector.load(DATASET)
total, valid, feasible = selector.counts(rows)
result = selector.compare(rows)
print('counts total=%d valid=%d feasible=%d' % (total, valid, feasible))
print('seed,selector_record,baseline_record,selector_mre,baseline_mre,difference,selector_acc,baseline_acc,selector_pred40,baseline_pred40')
with open(os.path.join(OUT, 'metrics.csv'), 'w', newline='', encoding='utf-8') as handle:
    fields = ['step','treatment','trial','domain','skill','total_loc','new_loc','test_pass','runtime_seconds','error_count','notes']
    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
    writer.writerow({'step':10,'treatment':5,'trial':2,'domain':'model configuration and quality','skill':'race','total_loc':0,'new_loc':0,'test_pass':1,'runtime_seconds':0,'error_count':0,'notes':'populated after measured run'})
for item in result['records']:
    s, b = item['selector'], item['baseline']
    print('%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g' % (item['seed'], s['original_record'], b['original_record'], s['MRE-'], b['MRE-'], item['difference'], s['ACC+'], b['ACC+'], s['PRED40+'], b['PRED40+']))
print('mean_difference=%.17g median_difference=%.17g wins=%d ties=%d losses=%d' % (result['mean_difference'], result['median_difference'], result['wins'], result['ties'], result['losses']))
