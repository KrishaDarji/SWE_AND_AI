"""Repeated paired grouped holdouts against a training-mean effort baseline."""
import argparse
import csv
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from predict_effort import LIMITATIONS, feature_matrix, fit_training
from summarize import PREDICTORS, numeric_summary, read_projects

SEEDS = list(range(42, 52))
TIE_ABS_TOL = 1e-12


def outcome(model_mae, baseline_mae):
    """Absolute MAE tolerance only; negative model-minus-baseline favors model."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v < 0 for v in (model_mae, baseline_mae)):
        raise ValueError('MAEs must be finite nonnegative numbers')
    difference = model_mae - baseline_mae
    return 'tie' if abs(difference) <= TIE_ABS_TOL else 'win' if difference < 0 else 'loss'


def validate(rows, seeds):
    if not isinstance(seeds, (list, tuple)) or not seeds or any(
        isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= 2**32 - 1 for s in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be a nonempty unique list of uint32 integers')
    if not isinstance(rows, (list, tuple)) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError('Need a nonempty sequence of project dictionaries')
    eligible, indices = [], []
    for i, row in enumerate(rows):
        value = row.get('Effort-')
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Effort- must be finite nonnegative or None')
        eligible.append(row)
        indices.append(i)
    keys = [tuple(r) for r in feature_matrix(eligible).tolist()]
    mapping = {}
    groups = [mapping.setdefault(key, len(mapping)) for key in keys]
    if len(mapping) < 2:
        raise ValueError('Need at least two distinct eligible normalized predictor groups')
    return eligible, indices, groups, mapping


def compare(rows, seeds=None):
    seeds = SEEDS.copy() if seeds is None else seeds
    eligible, indices, groups, mapping = validate(rows, seeds)
    results = []
    for seed in seeds:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            np.zeros((len(eligible), 1)), groups=groups))
        training = [eligible[i] for i in train]
        evaluation = [eligible[i] for i in test]
        # Shared row selection is outside both timers. Each fit creates fresh state.
        start = perf_counter()
        model, _ = fit_training(training)
        predictions = model.predict(feature_matrix(evaluation)).tolist()
        model_seconds = perf_counter() - start
        start = perf_counter()
        baseline = float(np.mean([r['Effort-'] for r in training]))
        baseline_predictions = [baseline] * len(evaluation)
        baseline_seconds = perf_counter() - start
        actual = [r['Effort-'] for r in evaluation]
        model_mae = float(mean_absolute_error(actual, predictions))
        baseline_mae = float(mean_absolute_error(actual, baseline_predictions))
        results.append({
            'seed': seed,
            'split': {'training_indices': [indices[i] for i in train],
                      'evaluation_indices': [indices[i] for i in test],
                      'training_group_ids': sorted({groups[i] for i in train}),
                      'evaluation_group_ids': sorted({groups[i] for i in test}),
                      'training_groups': len({groups[i] for i in train}),
                      'evaluation_groups': len({groups[i] for i in test}),
                      'total_groups': len(mapping), 'total_samples': len(rows),
                      'eligible_samples': len(eligible), 'excluded_missing_targets': len(rows) - len(eligible),
                      'training_samples': len(train), 'evaluation_samples': len(test)},
            'held_out_predictions': [dict(row_index=indices[i], actual=a, predicted=p,
                                          baseline_predicted=b)
                                     for i, a, p, b in zip(test, actual, predictions, baseline_predictions)],
            'baseline_training_mean': baseline,
            'model_mae': model_mae, 'baseline_mae': baseline_mae,
            'paired_mae_difference': model_mae - baseline_mae,
            'outcome': outcome(model_mae, baseline_mae),
            'model_runtime_seconds': model_seconds, 'baseline_runtime_seconds': baseline_seconds,
        })
    encoder = model.named_steps['categorical'].get_params()
    encoder['dtype'] = str(encoder['dtype'])
    return {
        'target': 'Effort-', 'units': 'unspecified', 'seeds': list(seeds),
        'methods': {'model': 'Unchanged predict_effort.fit_training; fresh pipeline per repeat',
                    'baseline': 'Constant mean of exactly the same training rows; no target imputation'},
        'predictor_fields': PREDICTORS,
        'excluded_inputs': ['Effort-', 'Defects-', 'Kloc+', 'all X metadata', 'all unknown fields'],
        'model_settings': model.named_steps['forest'].get_params(),
        'preprocessing': {'normalization': 'Existing normalize_proposal; strip strings and normalize missing tokens',
                          'imputer': model.named_steps['missing'].get_params(), 'encoder': encoder,
                          'fit_scope': 'Training rows only; no tuning using race results'},
        'dependency_versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                                'numpy': np.__version__, 'scipy': scipy.__version__},
        'split_procedure': {'method': 'GroupShuffleSplit', 'test_group_fraction': 0.25,
                            'rounding': 'ceil(0.25 * number of distinct eligible groups)',
                            'index_convention': 'zero-based CSV data row excluding header',
                            'group_ids': 'first occurrence among eligible rows; normalized predictor tuple only',
                            'missing_targets': 'excluded before grouping; never imputed',
                            'estimator_random_state': 42, 'seeds_control': 'split only; model settings unchanged'},
        'group_identities': [{'group_id': gid, 'normalized_predictors': list(key),
                              'row_indices': [indices[i] for i, g in enumerate(groups) if g == gid]}
                             for key, gid in mapping.items()],
        'excluded_missing_target_indices': [i for i, r in enumerate(rows) if r.get('Effort-') is None],
        'runtime_scope': {'clock': 'time.perf_counter wall seconds',
                          'model': 'fresh fit_training (including its unused mean computation) plus evaluation feature construction and prediction',
                          'baseline': 'training target extraction, mean calculation, and constant prediction list',
                          'excluded': 'imports, IO, validation, shared grouping/splitting and row selection, actual extraction, MAE and report serialization',
                          'order': 'model then baseline; small timings are noisy and descriptive'},
        'per_seed_results': results,
        'descriptive_summary': {key: numeric_summary([r[key] for r in results]) for key in
                                ('model_mae', 'baseline_mae', 'paired_mae_difference',
                                 'model_runtime_seconds', 'baseline_runtime_seconds')},
        'observed_outcomes': {key: sum(r['outcome'] == key for r in results) for key in ('win', 'tie', 'loss')},
        'tie_rule': {'absolute_tolerance': TIE_ABS_TOL, 'relative_tolerance': 0,
                     'definition': 'tie if abs(model MAE - baseline MAE) <= tolerance; win if lower otherwise loss'},
        'difference_definition': 'model MAE minus baseline MAE; negative favors model descriptively',
        'statistical_analysis': {'status': 'pending',
                                 'reason': 'Project-required same comparison procedure not supplied among allowed inputs',
                                 'substitute_performed': False, 'superiority_claim': False},
        'limitations': LIMITATIONS + [
            'Repeated partitions overlap within Trial 1; these are not independent LLM implementation trials.',
            'Raw MAE differences and win/tie/loss counts do not establish statistical superiority.',
            'Descriptive summaries weight seeds equally; each seed MAE weights evaluation rows equally.',
            'MAE is not an individual proposal uncertainty interval; no automatic approval claims.',
            'Timing is environment dependent and not reproducible bit for bit; partitions and predictions are reproducible in this environment.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(read_projects(args.dataset), args.seeds)
        report['dataset_path'] = str(args.dataset.resolve())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('seed model_mae baseline_mae model_minus_baseline outcome model_seconds baseline_seconds')
        for r in report['per_seed_results']:
            print(r['seed'], *(r[k] for k in ('model_mae', 'baseline_mae', 'paired_mae_difference',
                                             'outcome', 'model_runtime_seconds', 'baseline_runtime_seconds')))
        print(json.dumps({k: report[k] for k in ('descriptive_summary', 'observed_outcomes', 'statistical_analysis')}, indent=2))
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
