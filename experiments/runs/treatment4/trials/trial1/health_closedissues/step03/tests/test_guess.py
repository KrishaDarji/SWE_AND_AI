"""Grouped split, validation, leakage, and independent evaluation checks."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from guess import (CONFIGURATION, TARGETS, feature_matrix, fit_evaluate,
                   predict_configuration, split_rows, validate_configuration)
from summarize import load_rows

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def fixture():
    return [dict(zip(CONFIGURATION, [i, 'a' if i % 2 else 'b',
                                     i % 3, i / 10, i + 2]),
                 **{'MRE-': i / 10 + j, 'ACC+': 1 - i / 20 + j, 'PRED40+': 99})
            for i in range(15) for j in range(2)]


class GuessTests(unittest.TestCase):
    def test_input_validation(self):
        self.assertTrue(all(v is None for v in validate_configuration({}).values()))
        clean = validate_configuration({'N_estimators': ' 10.0 ', 'criterion': ' a ', 'Max_depth': ' '})
        self.assertEqual(clean['N_estimators'], 10)
        self.assertEqual(clean['criterion'], 'a')
        self.assertIsNone(clean['Max_depth'])
        for value in [[], None, {'MRE-': 1}, {'ACC+': 1}, {'PRED40+': 1}, {'extra': 1},
                      {'criterion': 1}, {'N_estimators': True}, {'Max_depth': []},
                      {'Max_depth': 'bad'}, {'Max_depth': math.inf}, {'Max_depth': 'NaN'}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_configuration(value)

    def test_normalized_group_split(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            p = Path(directory) / 'data.csv'
            p.write_text('MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n'
                         '1,2,3,10, a ,1,0,4\n4,5,6,10.0,a,1.0,0.0,4.0\n1,2,3,20,b,2,0,5\n')
            _, rows = load_rows(p)
        train, heldout, groups = split_rows(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in heldout})
        self.assertEqual(len({groups[i] for i in heldout}), 1)
        with self.assertRaises(ValueError):
            split_rows(rows[:2])

    def test_reproducibility_and_independent_mae(self):
        rows = fixture()
        _, a = fit_evaluate(rows)
        _, b = fit_evaluate(rows)
        self.assertEqual(a, b)
        self.assertEqual(a['split']['group_count'], 15)
        self.assertEqual(a['split']['heldout_group_count'], 3)
        self.assertEqual(a['split']['heldout_row_count'], 6)
        for target, report in a['targets'].items():
            predictions = report['heldout_predictions']
            model_mae = sum(abs(x['actual'] - x['predicted']) for x in predictions) / len(predictions)
            mean = sum(rows[i-1][target] for i in report['training_row_ids']) / report['training_sample_size']
            baseline_mae = sum(abs(x['actual'] - mean) for x in predictions) / len(predictions)
            self.assertAlmostEqual(report['model_mae'], model_mae)
            self.assertAlmostEqual(report['baseline_mae'], baseline_mae)
            self.assertAlmostEqual(report['training_mean_baseline'], mean)

    def test_targets_cannot_affect_fit_or_predictions(self):
        rows = fixture()
        train, test, _ = split_rows(rows)
        changed = copy.deepcopy(rows)
        for i in test:
            changed[i]['MRE-'] = 100000
            changed[i]['ACC+'] = -100000
        for row in changed:
            row['PRED40+'] = -999999
        a, report_a = fit_evaluate(rows)
        b, report_b = fit_evaluate(changed)
        self.assertEqual(report_a['split'], report_b['split'])
        np.testing.assert_array_equal(feature_matrix(rows), feature_matrix(changed))
        for target in TARGETS:
            np.testing.assert_array_equal(a[target].predict(feature_matrix(rows)),
                                          b[target].predict(feature_matrix(rows)))
            self.assertEqual(report_a['targets'][target]['training_mean_baseline'],
                             report_b['targets'][target]['training_mean_baseline'])
        changed = copy.deepcopy(rows)
        for row in changed:
            row['ACC+'] = 10000
        c, _ = fit_evaluate(changed)
        np.testing.assert_array_equal(a['MRE-'].predict(feature_matrix(rows)),
                                      c['MRE-'].predict(feature_matrix(rows)))
        for model in a.values():
            self.assertEqual(model.n_features_in_, 5)
            used = [i for _, _, indices in model.named_steps['preprocess'].transformers_ for i in indices]
            self.assertEqual(sorted(used), list(range(5)))

    def test_missing_values_unseen_category_and_training_only_preprocessing(self):
        rows = fixture()
        for row in rows:
            if row['N_estimators'] == 1:
                row['criterion'] = None
            if row['N_estimators'] == 0:
                row['N_estimators'] = None
        train, test, _ = split_rows(rows)
        rows[train[2]]['MRE-'] = None
        rows[test[0]]['ACC+'] = None
        for i in test:
            rows[i]['N_estimators'] = 1000000
            rows[i]['criterion'] = 'heldout_only'
        models, report = fit_evaluate(rows)
        for target in TARGETS:
            ids = report['targets'][target]['training_row_ids']
            expected_median = float(np.median([rows[i-1]['N_estimators'] for i in ids
                                             if rows[i-1]['N_estimators'] is not None]))
            self.assertEqual(report['targets'][target]['learned_numeric_imputation'][0], expected_median)
            self.assertNotIn('heldout_only', report['targets'][target]['learned_criterion_categories'])
        self.assertEqual(report['targets']['MRE-']['training_sample_size'], len(train)-1)
        self.assertEqual(report['targets']['ACC+']['heldout_sample_size'], len(test)-1)
        for config in [{}, {'criterion': 'unseen'}, {'Max_depth': None}]:
            self.assertTrue(all(math.isfinite(x) for x in predict_configuration(models, config).values()))
        for row in rows:
            row['criterion'] = None
            row['N_estimators'] = None
        models, report = fit_evaluate(rows)
        self.assertEqual(report['targets']['MRE-']['learned_numeric_imputation'][0], 0)
        self.assertTrue(all(math.isfinite(x) for x in predict_configuration(models, {'criterion': 'unseen'}).values()))

    def test_missing_target_labels_require_train_and_test_samples(self):
        rows = fixture()
        train, test, _ = split_rows(rows)
        for indices in (train, test):
            changed = copy.deepcopy(rows)
            for i in indices:
                changed[i]['MRE-'] = None
            with self.assertRaises(ValueError):
                fit_evaluate(changed)

    def test_cli_rejects_objective_unknown_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            config = Path(directory) / 'config.json'
            output = Path(directory) / 'evaluation.json'
            for content in ['{"MRE-": 1}', '{"unknown": 1}', '{"criterion":"a","criterion":"b"}', '[]']:
                config.write_text(content)
                p = subprocess.run([sys.executable, str(ROOT / 'source/guess.py'), str(DATASET),
                                    '--configuration', str(config), '--evaluation', str(output)],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn('Error:', p.stderr)
                self.assertFalse(output.exists())


class ActualPredictionTests(unittest.TestCase):
    def test_actual_dataset_evaluation_and_prediction(self):
        _, rows = load_rows(DATASET)
        models, report = fit_evaluate(rows)
        split = report['split']
        self.assertEqual(split['row_count'], 10000)
        self.assertEqual(split['heldout_group_count'], math.ceil(.2 * split['group_count']))
        self.assertFalse({r['group_id'] for r in split['training']} &
                         {r['group_id'] for r in split['heldout']})
        self.assertEqual({r['row_id'] for r in split['training'] + split['heldout']}, set(range(1,10001)))
        for target, result in report['targets'].items():
            items = result['heldout_predictions']
            mean = sum(rows[i-1][target] for i in result['training_row_ids']) / result['training_sample_size']
            self.assertAlmostEqual(result['model_mae'], sum(abs(p['actual']-p['predicted']) for p in items)/len(items))
            self.assertAlmostEqual(result['baseline_mae'], sum(abs(p['actual']-mean) for p in items)/len(items))
            self.assertEqual([p['actual'] for p in items], [rows[i-1][target] for i in result['heldout_row_ids']])
        config = json.loads((ROOT / 'example_configuration.json').read_text())
        self.assertEqual(set(predict_configuration(models, config)), set(TARGETS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
