import sys
from source.estimator import evaluate
path = sys.argv[1]
r = evaluate(path)
print('predictors=', ','.join(r['predictors']))
print('train_count=', r['train_count'], 'test_count=', r['test_count'])
print('heldout_mae=%.6f baseline_mae=%.6f' % (r['mae'], r['baseline_mae']))
print('model_settings=', r['model_settings'])
