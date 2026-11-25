#!/usr/bin/env python3
"""Independent Treatment 3 race for the x264 measurement table."""
import argparse, csv, random, time
from pathlib import Path
from statistics import mean, stdev
from sklearn.ensemble import RandomForestRegressor

FEATURES = ['no_asm','no_8x8dct','no_cabac','no_deblock','no_fast_pskip','no_mbtree',
            'no_mixed_refs','no_weightb','rc_lookahead','rc_lookahead_20',
            'rc_lookahead_40','rc_lookahead_60','ref','ref_1','ref_5','ref_9']
TARGET = 'Performance-'

def load_dataset(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(c not in rows[0] for c in FEATURES + [TARGET]):
        raise ValueError('missing required columns')
    data = []
    for row in rows:
        x = [int(row[c]) for c in FEATURES]
        y = float(row[TARGET].strip())
        if any(v not in (0, 1) for v in x) or y <= 0:
            raise ValueError('invalid value')
        data.append((x, y))
    if len({tuple(x) for x, _ in data}) != len(data):
        raise ValueError('duplicate configurations')
    return data

def race(data, seed, budget=32, warmup=8, batch=8):
    rng = random.Random(seed)
    order = list(range(len(data))); rng.shuffle(order)
    split = int(.75 * len(order)); pool, holdout = order[:split], order[split:]
    evaluated = rng.sample(pool, warmup)
    remaining = [i for i in pool if i not in evaluated]
    while len(evaluated) < budget:
        model = RandomForestRegressor(n_estimators=100, random_state=seed + len(evaluated),
                                      min_samples_leaf=2, n_jobs=1)
        model.fit([data[i][0] for i in evaluated], [data[i][1] for i in evaluated])
        take = min(batch, budget - len(evaluated))
        scored = sorted(remaining, key=lambda i: model.predict([data[i][0]])[0])
        chosen = scored[:take]
        evaluated.extend(chosen)
        remaining = [i for i in remaining if i not in set(chosen)]
    best = min(data[i][1] for i in evaluated)
    optimum = min(y for _, y in data)
    return best, optimum, len(evaluated)

def random_baseline(data, seed, budget=32):
    rng = random.Random(seed)
    split = int(.75 * len(data)); order = list(range(len(data))); rng.shuffle(order)
    chosen = order[:budget]
    best = min(data[i][1] for i in chosen)
    return best, min(y for _, y in data), budget

def run(path, repeats=20, budget=32):
    data = load_dataset(path)
    records = []
    for method, fn in [('treatment3_selector', race), ('random_baseline', random_baseline)]:
        for seed in range(repeats):
            start = time.perf_counter(); best, optimum, evaluations = fn(data, seed, budget)
            records.append({'method': method, 'seed': seed, 'best_performance': best,
                            'regret_percent': 100*(best-optimum)/optimum,
                            'evaluations': evaluations, 'runtime_seconds': time.perf_counter()-start})
    summary = {}
    for method in ('treatment3_selector', 'random_baseline'):
        rows = [r for r in records if r['method'] == method]
        summary[method] = {'mean_regret_percent': mean(r['regret_percent'] for r in rows),
                           'stdev_regret_percent': stdev(r['regret_percent'] for r in rows),
                           'mean_best_performance': mean(r['best_performance'] for r in rows),
                           'mean_runtime_seconds': mean(r['runtime_seconds'] for r in rows)}
    return data, records, summary

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--repeats', type=int, default=20); ap.add_argument('--budget', type=int, default=32)
    args = ap.parse_args(); data, records, summary = run(args.dataset, args.repeats, args.budget)
    print(f'rows={len(data)} features={len(FEATURES)} repeats={args.repeats} budget={args.budget}')
    for m, s in summary.items():
        print(f'{m}: mean_regret_percent={s["mean_regret_percent"]:.3f} stdev={s["stdev_regret_percent"]:.3f} mean_best={s["mean_best_performance"]:.3f} mean_runtime_seconds={s["mean_runtime_seconds"]:.4f}')
    winner = min(summary, key=lambda m: summary[m]['mean_regret_percent'])
    print('winner=' + winner)
    return records

if __name__ == '__main__': main()
