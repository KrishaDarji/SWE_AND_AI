"""Training-only categorical effort estimation with a grouped held-out evaluation."""
import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from summarize import MISSING, PREDICTORS, read_projects

SEED = 42
LIMITATIONS = [
    'Effort- physical units are unspecified; none are inferred.',
    'An effort estimate alone does not justify approving a proposal.',
    'Small historical observational dataset; one grouped split is not a guarantee of future accuracy.',
    'No uncertainty intervals, approval guarantees, causal claims, or held-out tuning.',
    'No proposal-time size estimate is supplied; delivered Kloc+ is excluded.',
    'Predictor availability at proposal time is assumed and must be checked operationally.',
    'Unseen categories encode as all zeros; missing categories use an explicit empty-string level.',
    'Rows missing Effort- are excluded, never target-imputed; reported MAE is row-weighted.',
    'Group split prevents identical normalized inputs crossing partitions, but related projects may remain.',
    'Random forest predictions average training outcomes and may miss extreme or novel efforts.',
    'Evaluation model remains fitted only to training rows; no full-dataset refit is performed.',
]


def normalize_proposal(proposal):
    """Allow known predictor keys only; omitted/null/missing-token values are missing."""
    if not isinstance(proposal, dict):
        raise ValueError('Proposal must be a JSON object')
    extra = set(proposal) - set(PREDICTORS)
    if extra:
        raise ValueError('Forbidden or unknown proposal fields: ' + ', '.join(sorted(extra)))
    values = []
    for field in PREDICTORS:
        value = proposal.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'{field}: expected a categorical string or null')
        value = value.strip() if value is not None else None
        values.append(None if value is None or value in MISSING else value)
    return values


def feature_matrix(rows):
    return np.array([normalize_proposal({p: row.get(p) for p in PREDICTORS}) for row in rows], dtype=object)


def split_projects(rows):
    """Hold out 25% (rounded up) of unique input groups, without inspecting outcomes."""
    keys = [tuple(row) for row in feature_matrix(rows).tolist()]
    group_numbers = {}
    groups = [group_numbers.setdefault(key, len(group_numbers)) for key in keys]
    if len(group_numbers) < 2:
        raise ValueError('Need at least two distinct predictor configurations')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(
        np.zeros((len(rows), 1)), groups=groups))
    return train.tolist(), test.tolist(), groups


def fit_training(rows):
    """Only the provided training rows enter learned preprocessing and model fitting."""
    if not rows:
        raise ValueError('Training rows must not be empty')
    targets = [row.get('Effort-') for row in rows]
    if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in targets):
        raise ValueError('Training efforts must be finite nonnegative numbers')
    pipeline = Pipeline([
        ('missing', SimpleImputer(missing_values=None, strategy='constant', fill_value='', keep_empty_features=True)),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('forest', RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=1)),
    ])
    pipeline.fit(feature_matrix(rows), targets)
    return pipeline, float(np.mean(targets))


def estimate(model, proposal):
    return float(model.predict(np.array([normalize_proposal(proposal)], dtype=object))[0])


def evaluate(rows):
    eligible = []
    original_indices = []
    for index, row in enumerate(rows):
        value = row.get('Effort-')
        if value is None:
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Effort- must be finite and nonnegative when present')
        eligible.append(row)
        original_indices.append(index)
    train, test, groups = split_projects(eligible)
    model, baseline = fit_training([eligible[i] for i in train])
    predicted = model.predict(feature_matrix([eligible[i] for i in test]))
    actual = [eligible[i]['Effort-'] for i in test]
    report = {
        'target': 'Effort-', 'predictor_fields': PREDICTORS,
        'excluded_inputs': ['Effort-', 'Defects-', 'Kloc+', 'all X-suffixed metadata', 'all unknown fields'],
        'preprocessing': {'normalization': 'strip strings; Step 1 missing tokens and omitted/null become None',
                          'imputer': model.named_steps['missing'].get_params(),
                          'encoder': model.named_steps['categorical'].get_params(),
                          'fit_scope': 'training rows only; no ordinal mapping'},
        'model_settings': model.named_steps['forest'].get_params(),
        'dependency_versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                                'numpy': np.__version__, 'scipy': scipy.__version__},
        'split': {'method': 'GroupShuffleSplit; normalized predictor tuples define groups',
                  'seed': SEED, 'test_group_fraction': 0.25, 'n_splits': 1,
                  'index_convention': 'zero-based data row indices, excludes CSV header',
                  'total_samples': len(rows), 'eligible_samples': len(eligible),
                  'excluded_missing_targets': len(rows) - len(eligible),
                  'training_samples': len(train), 'evaluation_samples': len(test),
                  'total_groups': len(set(groups)),
                  'training_groups': len({groups[i] for i in train}),
                  'evaluation_groups': len({groups[i] for i in test}),
                  'training_indices': [original_indices[i] for i in train],
                  'evaluation_indices': [original_indices[i] for i in test]},
        'baseline': {'strategy': 'training-set mean', 'prediction': baseline},
        'baseline_mae': float(mean_absolute_error(actual, [baseline] * len(test))),
        'model_mae': float(mean_absolute_error(actual, predicted)),
        'held_out_predictions': [{'row_index': original_indices[i], 'actual': truth,
                                  'predicted': float(pred), 'baseline_predicted': baseline}
                                 for i, truth, pred in zip(test, actual, predicted)],
        'limitations': LIMITATIONS,
    }
    # dtype is a Python class in sklearn's encoder params; record its explicit name.
    report['preprocessing']['encoder']['dtype'] = str(report['preprocessing']['encoder']['dtype'])
    return model, report


def load_proposal(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    with Path(path).open(encoding='utf-8') as stream:
        proposal = json.load(stream, object_pairs_hook=unique_object)
    normalize_proposal(proposal)
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--proposal', required=True, type=Path, help='JSON object containing predictor fields only')
    parser.add_argument('--evaluation', required=True, type=Path)
    args = parser.parse_args()
    try:
        proposal = load_proposal(args.proposal)
        model, report = evaluate(read_projects(args.dataset))
        report['dataset_path'] = str(args.dataset.resolve())
        prediction = estimate(model, proposal)
        args.evaluation.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'estimated Effort-': prediction, 'units': 'unspecified',
                          'training_samples': report['split']['training_samples'],
                          'evaluation_samples': report['split']['evaluation_samples'],
                          'model_mae': report['model_mae'], 'baseline_mae': report['baseline_mae'],
                          'decision_note': 'An effort estimate alone does not justify approving a proposal.'}, indent=2))
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')


if __name__ == '__main__':
    main()
