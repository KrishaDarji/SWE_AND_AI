"""Predict recorded Health MRE- and ACC+ from configuration fields only."""
import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from summarize import CONFIGURATION, OBJECTIVES, load_rows

TARGETS = {'MRE-': 'minimize', 'ACC+': 'maximize'}
NUMERIC = [i for i, name in enumerate(CONFIGURATION) if name != 'criterion']
SETTINGS = dict(n_estimators=120, max_depth=12, min_samples_leaf=2,
                max_features=1.0, bootstrap=True, criterion='squared_error',
                random_state=42, n_jobs=1)
LIMITATIONS = [
    'Targets are recorded CSV metrics; no metric formula, unit, cost, or causal meaning is inferred.',
    'A single seed-42 group holdout estimates performance on unseen configurations from this dataset only.',
    'No tuning uses held-out targets; model settings are fixed in advance.',
    'Unseen categories encode as all zero; unseen numeric values may extrapolate poorly.',
    'Missing numeric inputs use training medians (zero for entirely missing columns); missing criterion uses the training mode (zero category if entirely missing).',
    'Missing target labels are excluded separately per target; neither targets nor PRED40+ enter predictors.',
    'No theoretical metric ranges are assumed or imposed. Forest predictions are within the training target range; this does not guarantee validity on unseen configurations.',
    'Predictions are point estimates with no uncertainty or accuracy guarantee; evaluation MAE is in each recorded target scale.',
    'Numeric inputs follow Step 1 finite-float validation; no undocumented parameter-domain or integer restrictions are added.',
    'Row IDs are one-based data-record positions, not persistent identifiers; header is CSV line 1.'
]


def validate_configuration(value):
    """Require only known predictor keys; omitted/null/blank fields mean missing."""
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object')
    extra = set(value) - set(CONFIGURATION)
    if extra:
        raise ValueError(f'Objective or unknown configuration fields: {sorted(extra)}')
    result = {}
    for name in CONFIGURATION:
        cell = value.get(name)
        if cell is None or (isinstance(cell, str) and not cell.strip()):
            result[name] = None
        elif name == 'criterion':
            if not isinstance(cell, str):
                raise ValueError('criterion must be a string or null')
            result[name] = cell.strip()
        else:
            if isinstance(cell, bool) or not isinstance(cell, (str, int, float)):
                raise ValueError(f'{name} must be a finite number or null')
            try:
                number = float(cell)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{name} must be a finite number') from exc
            if not math.isfinite(number):
                raise ValueError(f'{name} must be finite')
            result[name] = number
    return result


def feature_matrix(rows):
    """Select only the five predictors, regardless of other keys in CSV rows."""
    return np.asarray([[row[name] if row[name] is not None else
                        (None if name == 'criterion' else np.nan)
                        for name in CONFIGURATION] for row in rows], dtype=object)


def make_model():
    numeric = SimpleImputer(strategy='median', keep_empty_features=True)
    category = Pipeline([
        ('impute', SimpleImputer(missing_values=None, strategy='most_frequent',
                                 keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocess = ColumnTransformer([
        ('numeric', numeric, NUMERIC), ('category', category, [1])], remainder='drop')
    return Pipeline([('preprocess', preprocess),
                     ('regressor', RandomForestRegressor(**SETTINGS))])


def split_rows(rows):
    """Group on normalized predictors alone, holding out ceil(20% of groups)."""
    mapping = {}
    groups = []
    for row in rows:
        key = tuple(row[name] for name in CONFIGURATION)
        if key not in mapping:
            mapping[key] = len(mapping)
        groups.append(mapping[key])
    if len(mapping) < 2:
        raise ValueError('At least two distinct normalized configurations required')
    train, heldout = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                            random_state=42).split(rows, groups=groups))
    return train, heldout, groups


def fit_evaluate(rows):
    """Fit per-target pipelines on labeled training rows; evaluate only afterward."""
    train, heldout, groups = split_rows(rows)
    features = feature_matrix(rows)
    models, results = {}, {}
    for target in TARGETS:
        labeled_train = [int(i) for i in train if rows[i][target] is not None]
        labeled_test = [int(i) for i in heldout if rows[i][target] is not None]
        if not labeled_train or not labeled_test:
            raise ValueError(f'{target}: labeled training and held-out rows required')
        y_train = np.asarray([rows[i][target] for i in labeled_train])
        model = make_model().fit(features[labeled_train], y_train)
        baseline = float(np.mean(y_train))
        predicted = model.predict(features[labeled_test])
        actual = [rows[i][target] for i in labeled_test]
        pre = model.named_steps['preprocess']
        results[target] = {
            'direction': TARGETS[target], 'training_sample_size': len(labeled_train),
            'heldout_sample_size': len(labeled_test),
            'training_row_ids': [i + 1 for i in labeled_train],
            'heldout_row_ids': [i + 1 for i in labeled_test],
            'training_mean_baseline': baseline,
            'training_target_range': [float(min(y_train)), float(max(y_train))],
            'model_mae': float(mean_absolute_error(actual, predicted)),
            'baseline_mae': float(mean_absolute_error(actual, np.full(len(actual), baseline))),
            'learned_numeric_imputation': pre.named_transformers_['numeric'].statistics_.tolist(),
            'learned_criterion_categories': pre.named_transformers_['category'].named_steps['encode'].categories_[0].tolist(),
            'heldout_predictions': [dict(row_id=i + 1, actual=y, predicted=float(p), baseline=baseline)
                                    for i, y, p in zip(labeled_test, actual, predicted)]}
        models[target] = model
    identity = lambda indices: [dict(row_id=int(i) + 1, csv_line=int(i) + 2,
                                     group_id=groups[i]) for i in indices]
    split = {'seed': 42, 'test_group_fraction': .2,
             'method': 'GroupShuffleSplit on first-seen IDs of normalized predictor tuples',
             'row_count': len(rows), 'group_count': len(set(groups)),
             'training_row_count': len(train), 'heldout_row_count': len(heldout),
             'training_group_count': len({groups[i] for i in train}),
             'heldout_group_count': len({groups[i] for i in heldout}),
             'training': identity(train), 'heldout': identity(heldout),
             'groups': [dict(group_id=g, configuration={n: r[n] for n in CONFIGURATION})
                        for g, r in sorted({groups[i]: r for i, r in enumerate(rows)}.items())]}
    return models, {'target_definitions': {t: {'direction': d, 'definition': 'Recorded CSV column; formula unspecified'} for t, d in TARGETS.items()},
                    'predictor_fields': list(CONFIGURATION), 'excluded_objectives': list(OBJECTIVES),
                    'model_settings': {'estimator': 'Independent RandomForestRegressor per target', **SETTINGS},
                    'preprocessing': {'numeric_fields': [CONFIGURATION[i] for i in NUMERIC],
                                      'numeric': 'Training median; keep empty columns with zero fallback',
                                      'criterion': 'Training mode imputation then one-hot; unknown categories ignored',
                                      'fit_scope': 'Labeled training rows separately for each target'},
                    'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                                 'numpy': np.__version__, 'scipy': scipy.__version__},
                    'split': split, 'targets': results, 'limitations': LIMITATIONS}


def predict_configuration(models, configuration):
    clean = validate_configuration(configuration)
    features = feature_matrix([clean])
    return {target: float(models[target].predict(features)[0]) for target in TARGETS}


def unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate JSON field: {name}')
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        config = validate_configuration(json.loads(args.configuration.read_text(),
                                                  object_pairs_hook=unique_json_object))
        _, rows = load_rows(args.dataset)
        models, report = fit_evaluate(rows)
        report['dataset_path'] = str(args.dataset.resolve())
        prediction = predict_configuration(models, config)
        report['illustrative_demo'] = {'label': 'Illustrative configuration; not an optimized recommendation',
                                       'configuration': config, 'predictions': prediction}
        args.evaluation.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(json.dumps({'label': report['illustrative_demo']['label'], 'predictions': prediction,
                      'heldout_mae': {t: {'model': r['model_mae'], 'training_mean_baseline': r['baseline_mae']}
                                      for t, r in report['targets'].items()}}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
