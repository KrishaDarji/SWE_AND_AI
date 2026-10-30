import sys
sys.path.insert(0, 'source')
from certify import main, load_groups, _mean_var
data = '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'
threshold = 0.10 * _mean_var(load_groups(data)['n'])[0]
print('Illustrative threshold: 10% of tool=n mean effort (not a client requirement) =', threshold)
main(['certify.py', data, str(threshold)])
