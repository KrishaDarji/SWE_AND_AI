#!/usr/bin/env python3
"""Independent categorical estimator and evidence report for NASA93."""
import csv, math, random, statistics

DESCRIPTORS = ['prec','flex','resl','team','pmat','rely','data','cplx','ruse','docu','time','stor','pvol','acap','pcap','pcon','apex','plex','ltex','tool','site','sced']
TARGET = 'Effort-'

def clean(value):
    return ' '.join(str(value).strip().split()).lower()

def read_rows(path):
    with open(path, newline='') as f:
        rows = []
        for original, row in enumerate(csv.DictReader(f)):
            try:
                target = float(row[TARGET])
                if not math.isfinite(target):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({'original_id': row.get('idX', str(original + 1)), 'row_number': original + 2,
                         'x': tuple(clean(row.get(k, '')) for k in DESCRIPTORS), 'y': target})
    return rows

def split_grouped(rows, seed=42):
    groups = {}
    for row in rows:
        groups.setdefault(row['x'], []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    cut = max(1, min(len(keys) - 1, round(len(keys) * .25))) if len(keys) > 1 else 0
    test_keys = set(keys[:cut])
    train = [r for r in rows if r['x'] not in test_keys]
    test = [r for r in rows if r['x'] in test_keys]
    return train, test

def median_reference(train):
    if not train: raise ValueError('training data is empty')
    return statistics.median(r['y'] for r in train)

def fit(train):
    if len({r['x'] for r in train}) < 1: raise ValueError('insufficient descriptor groups')
    return {'rows': list(train), 'reference': median_reference(train)}

def predict(model, proposal):
    if not isinstance(proposal, dict): raise ValueError('proposal must be a mapping')
    missing = [k for k in DESCRIPTORS if k not in proposal]
    if missing: raise ValueError('missing descriptors: ' + ','.join(missing))
    x = tuple(clean(proposal[k]) for k in DESCRIPTORS)
    if any(v == '' for v in x): raise ValueError('proposal contains blank descriptor')
    exact = [r for r in model['rows'] if r['x'] == x]
    if exact:
        value = statistics.median(r['y'] for r in exact)
        return value, {'kind':'exact training match', 'matches': [r['original_id'] for r in exact]}
    distances = []
    for r in model['rows']:
        distance = sum(a != b for a, b in zip(x, r['x']))
        distances.append((distance, str(r['original_id']), r))
    distances.sort(key=lambda z: (z[0], z[1]))
    nearest = distances[:min(5, len(distances))]
    value = sum(z[2]['y'] for z in nearest) / len(nearest)
    return value, {'kind':'5-nearest categorical neighbors', 'matches':[z[2]['original_id'] for z in nearest], 'distances':[z[0] for z in nearest]}

def associations(train):
    out = []
    for i, name in enumerate(DESCRIPTORS):
        groups = {}
        for r in train: groups.setdefault(r['x'][i], []).append(r['y'])
        if len(groups) < 2: continue
        means = {k: statistics.mean(v) for k,v in groups.items()}
        ordered = sorted(means.items(), key=lambda z: (z[1], z[0]))
        out.append({'descriptor':name, 'lowest':ordered[0][0], 'lowest_mean':ordered[0][1], 'highest':ordered[-1][0], 'highest_mean':ordered[-1][1], 'label':'training-only historical association'})
    return out

def explain(model, proposal, prediction, detail):
    return {'estimate':prediction, 'reference':model['reference'], 'reference_label':'training-only median Effort-', 'behavior_label':detail['kind'], 'model_matches':detail['matches'], 'historical_associations':associations(model['rows']), 'scope_note':'Associations are global training summaries, not additive local contributions or causal effects.'}

def run(csv_path):
    rows = read_rows(csv_path)
    train, test = split_grouped(rows)
    if not train or not test: raise ValueError('insufficient descriptor groups for split')
    model = fit(train)
    held = min(test, key=lambda r: r['row_number'])
    proposal = dict(zip(DESCRIPTORS, held['x']))
    estimate, detail = predict(model, proposal)
    frozen = explain(model, proposal, estimate, detail)
    print('TRAIN_ROWS', len(train)); print('TEST_ROWS', len(test)); print('HELD_OUT_ID', held['original_id'])
    print('ESTIMATE', frozen['estimate']); print('REFERENCE', frozen['reference'])
    print('EXPLANATION', frozen['behavior_label'], frozen['model_matches'])
    print('AUDIT_OBSERVED_EFFORT', held['y'])
    return frozen, held

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2: raise SystemExit('usage: justify_estimator.py CSV_PATH')
    run(sys.argv[1])
