"""Equal-budget descriptive comparison; full-data audit follows all frozen searches."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import sklearn

from choose import (Candidate, CSVOracle, QUALITY_LIMITATION, STRATEGY,
                    load_candidates, positive_integer, post_selection_audit, select)
from flow import Flow, _number
from remember import OBJECTIVE, describe

METHODS = ('step7_selector', 'uniform_random')
RUNTIME_SCOPE = ('perf_counter wall seconds around selector invocation only, including '
                 'initialization, oracle CSV reads, model fitting, acquisition and history; '
                 'excludes imports, shared settings-only loading, oracle construction, '
                 'serialization, reporting and full-data audit. Methods run serially, '
                 'selector then random for each seed; timings are environment dependent.')
LIMITATIONS = [
    'Performance- is minimized; physical units are unspecified.',
    QUALITY_LIMITATION,
    'Seeds repeat searches within Trial 1; they are not independent LLM implementation trials.',
    'Candidate identity is the original CSV data row; duplicate settings remain separate rows.',
    'This offline oracle replays measured outcomes; it does not execute a video encoder.',
    'Observed differences and win counts do not establish statistical superiority.',
    'The required same comparison procedure is absent from allowed inputs; analysis is pending.',
    'The API objective boundary is not a hostile-code Python sandbox.',
    'Runtime includes CSV replay overhead and fixed method order; it is not encoder runtime.',
]


def validate(settings, candidates, budget, seeds):
    """Validate the entire requested experiment before any counted objective access."""
    positive_integer(budget, 'budget')
    settings, candidates, seeds = tuple(settings), tuple(candidates), tuple(seeds)
    Flow(settings)
    if not candidates or budget > len(candidates):
        raise ValueError('budget must not exceed the nonempty candidate count')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and distinct')
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
            raise ValueError('seed must be an integer in [0, 2**32)')
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise ValueError('Expected settings-only Candidate records')
        positive_integer(candidate.data_row, 'data_row')
        if len(candidate.values) != len(settings):
            raise ValueError('Candidate settings width does not match schema')
        for value in candidate.values:
            _number(value)
        ids.append(candidate.data_row)
    if len(set(ids)) != len(ids):
        raise ValueError('Candidate data rows must be distinct')
    return settings, tuple(sorted(candidates, key=lambda c: c.data_row)), seeds


def uniform_search(settings, candidates, evaluate, *, budget, seed):
    """Uniform row sampling without replacement; first observed minimum wins ties."""
    settings, candidates, _ = validate(settings, candidates, budget, (seed,))
    order = random.Random(seed).sample(candidates, budget)
    history, best = [], None
    for step, candidate in enumerate(order, 1):
        value = _number(evaluate(candidate.data_row))
        config = dict(zip(settings, candidate.values))
        if best is None or value < best[OBJECTIVE]:
            best = {'data_row': candidate.data_row, 'settings': config, OBJECTIVE: value}
        history.append({'evaluation': step, 'data_row': candidate.data_row,
                        'settings': config, OBJECTIVE: value,
                        'reason': 'uniform without replacement',
                        'best_so_far': {**best, 'settings': dict(best['settings'])}})
    return {'budget': budget, 'seed': seed, 'candidate_count': len(candidates),
            'actual_evaluation_count': len(history), 'evaluation_history': history,
            'selected_candidate': best}


def search_runs(settings, candidates, budget, seeds, oracle_factory):
    """Only a factory of fresh counted callables supplies objective access; no audit."""
    settings, candidates, seeds = validate(settings, candidates, budget, seeds)
    runs = []
    for seed in seeds:
        for method, search in zip(METHODS, (select, uniform_search)):
            oracle = oracle_factory(budget)
            if oracle.count != 0:
                raise ValueError('Oracle must start fresh with zero evaluations')
            start = time.perf_counter()
            result = search(settings, candidates, oracle, budget=budget, seed=seed)
            elapsed = time.perf_counter() - start
            if oracle.count != budget or result['actual_evaluation_count'] != budget:
                raise RuntimeError('Equal-budget enforcement failed')
            history = result['evaluation_history']
            if len(history) != budget or len({h['data_row'] for h in history}) != budget:
                raise RuntimeError('Expected exactly budget distinct observed rows')
            runs.append({'method': method, 'seed': seed, 'budget': budget,
                         'evaluation_count': oracle.count, 'search_runtime_seconds': elapsed,
                         'evaluation_order': [h['data_row'] for h in history],
                         'evaluation_history': history,
                         'selected_candidate': result['selected_candidate']})
    return runs


def summarize(runs, audits):
    """Descriptive summaries only; exact numeric equality defines paired ties."""
    summaries = {}
    for method in METHODS:
        indices = [i for i, run in enumerate(runs) if run['method'] == method]
        summaries[method] = {
            'selected_performance': describe([runs[i]['selected_candidate'][OBJECTIVE] for i in indices]),
            'evaluation_count': describe([runs[i]['evaluation_count'] for i in indices]),
            'absolute_gap': describe([audits[i]['absolute_gap'] for i in indices]),
            'search_runtime_seconds': describe([runs[i]['search_runtime_seconds'] for i in indices]),
        }
    paired, counts = [], {'wins': 0, 'ties': 0, 'losses': 0}
    seeds = list(dict.fromkeys(run['seed'] for run in runs))
    for seed in seeds:
        values = {run['method']: run['selected_candidate'][OBJECTIVE]
                  for run in runs if run['seed'] == seed}
        delta = values[METHODS[0]] - values[METHODS[1]]
        outcome = 'ties' if delta == 0 else 'wins' if delta < 0 else 'losses'
        counts[outcome] += 1
        paired.append({'seed': seed, 'selector_minus_random': delta, 'selector_outcome': outcome})
    return {'methods': summaries, 'paired_observed': {
        'reference': METHODS[0], 'tie_rule': 'Exact numeric equality; no tolerance.',
        **counts, 'per_seed': paired},
        'interpretation': 'Descriptive observations only; no statistical superiority claim.'}


def run_race(dataset, output_dir, *, budget=20, seeds=tuple(range(42, 52))):
    """Freeze every selection to disk before any full-data audit is invoked."""
    dataset, output_dir = Path(dataset), Path(output_dir)
    targets = [output_dir / name for name in ('frozen_selections.json', 'race.json', 'per_seed.csv')]
    if any(target.resolve() == dataset.resolve() for target in targets):
        raise ValueError('Output must not overwrite the dataset')
    settings, candidates = load_candidates(dataset)
    settings, candidates, seeds = validate(settings, candidates, budget, seeds)
    runs = search_runs(settings, candidates, budget, seeds, lambda b: CSVOracle(dataset, b))
    frozen = json.dumps(runs, sort_keys=True, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(frozen + '\n', encoding='utf-8')
    # All methods and all seeds have finished. Audits only receive frozen copies.
    audits = []
    for run in json.loads(frozen):
        audit = post_selection_audit(dataset, json.dumps(run, sort_keys=True, allow_nan=False))
        audits.append({'method': run['method'], 'seed': run['seed'], **audit})
    result = {
        'dataset': str(dataset.resolve()), 'candidate_count': len(candidates),
        'features': list(settings), 'objective': OBJECTIVE, 'direction': 'minimize', 'units': None,
        'methods': {METHODS[0]: STRATEGY, METHODS[1]: {
            'name': 'Uniform random search without replacement',
            'sampling': 'random.Random(seed).sample over rows in CSV order',
            'observed_ties': 'First evaluated minimum retained'}},
        'budget': budget, 'budgets': {method: budget for method in METHODS}, 'seeds': list(seeds),
        'runs': json.loads(frozen), 'audit_results': audits,
        'freeze': {'file': targets[0].name, 'all_searches_finished_before_audit': True,
                   'sha256': hashlib.sha256(frozen.encode()).hexdigest()},
        'descriptive_summaries': summarize(json.loads(frozen), audits),
        'runtime_scope': RUNTIME_SCOPE,
        'statistical_analysis': {'status': 'pending',
            'reason': 'Project required same comparison procedure was not supplied among allowed inputs.',
            'substitute_test_performed': False, 'statistical_superiority_claimed': False},
        'quality_constraint_verified': False, 'limitations': LIMITATIONS,
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__},
    }
    targets[1].write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with targets[2].open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'seed', 'budget', 'evaluation_count',
            'selected_data_row', OBJECTIVE, 'absolute_gap', 'search_runtime_seconds'])
        writer.writeheader()
        for run, audit in zip(result['runs'], audits):
            writer.writerow({k: run[k] for k in ('method', 'seed', 'budget', 'evaluation_count',
                             'search_runtime_seconds')} | {
                'selected_data_row': run['selected_candidate']['data_row'],
                OBJECTIVE: run['selected_candidate'][OBJECTIVE], 'absolute_gap': audit['absolute_gap']})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 52)))
    args = parser.parse_args(argv)
    try:
        result = run_race(args.dataset, args.output_dir, budget=args.budget, seeds=args.seeds)
    except (OSError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')
    print('method,seed,budget,evaluations,selected_Performance-,gap,search_seconds')
    for run, audit in zip(result['runs'], result['audit_results']):
        print(f"{run['method']},{run['seed']},{run['budget']},{run['evaluation_count']},"
              f"{run['selected_candidate'][OBJECTIVE]},{audit['absolute_gap']},"
              f"{run['search_runtime_seconds']:.9f}")
    print(json.dumps(result['descriptive_summaries'], indent=2))
    print(RUNTIME_SCOPE)
    print('Statistical analysis: pending; required procedure not supplied.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
