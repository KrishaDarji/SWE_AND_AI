"""Step 2 validation, partition isolation, and independent evaluation checks."""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from predict_effort import (PREDICTORS, estimate, evaluate, feature_matrix, fit_training,
                            load_proposal, normalize_proposal, split_projects)
from summarize import read_projects

DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


def fixture():
    return [dict.fromkeys(PREDICTORS, f'category_{i}') | {'Effort-': float(i * 10 + 1)} for i in range(12)]


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_projects(DATASET)
        cls.model, cls.report = evaluate(cls.rows)

    def test_proposal_validation(self):
        for field in ['Effort-', 'Defects-', 'Kloc+', 'idX', 'centerX', 'YearX', 'MonthsX', 'futureX', 'typo']:
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_proposal({field: 1})
        for value in [[], None, 1, 'n', True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_proposal(value)
        for value in [1, True, [], {}, float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_proposal({'prec': value})

    def test_whitespace_missing_and_unseen(self):
        proposal = dict.fromkeys(PREDICTORS, ' n\t')
        self.assertEqual(normalize_proposal(proposal), ['n'] * len(PREDICTORS))
        self.assertEqual(estimate(self.model, proposal), estimate(self.model, dict.fromkeys(PREDICTORS, 'n')))
        for value in [None, '', ' ? ', 'NA', ' N/A ', 'null']:
            self.assertEqual(normalize_proposal({'prec': value}), [None] * len(PREDICTORS))
            self.assertTrue(math.isfinite(estimate(self.model, {'prec': value})))
        novel = dict.fromkeys(PREDICTORS, 'never_seen_category')
        self.assertTrue(math.isfinite(estimate(self.model, novel)))
        encoded = self.model[:-1].transform(feature_matrix([novel]))
        np.testing.assert_array_equal(encoded, np.zeros_like(encoded))

    def test_missing_training_column_and_missing_target(self):
        rows = fixture()
        for row in rows:
            row['prec'] = None
        rows[0]['flex'] = ' ? '
        model, report = evaluate(rows)
        self.assertTrue(math.isfinite(estimate(model, {})))
        self.assertIn('', model.named_steps['categorical'].categories_[0])
        rows[0]['Effort-'] = None
        _, report = evaluate(rows)
        self.assertEqual(report['split']['excluded_missing_targets'], 1)
        self.assertNotIn(0, report['split']['training_indices'] + report['split']['evaluation_indices'])

    def test_group_partition_and_reproducibility(self):
        train, test, groups = split_projects(self.rows)
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})
        self.assertEqual(set(train) | set(test), set(range(len(self.rows))))
        self.assertEqual(split_projects(self.rows), (train, test, groups))
        model, report = evaluate(self.rows)
        self.assertEqual(report, self.report)
        self.assertEqual(estimate(model, {}), estimate(self.model, {}))
        duplicate = copy.deepcopy(fixture())
        duplicate.append(dict(duplicate[0], **{'Effort-': 999}))
        duplicate[-1]['prec'] = ' category_0 '
        train, test, groups = split_projects(duplicate)
        self.assertEqual(groups[0], groups[-1])
        self.assertEqual(0 in train, len(duplicate) - 1 in train)
        self.assertEqual(0 in test, len(duplicate) - 1 in test)

    def test_independent_mae_and_training_mean(self):
        report = self.report
        records = report['held_out_predictions']
        baseline = sum(self.rows[i]['Effort-'] for i in report['split']['training_indices']) / report['split']['training_samples']
        self.assertAlmostEqual(report['baseline']['prediction'], baseline)
        for key, column in [('model_mae', 'predicted'), ('baseline_mae', 'baseline_predicted')]:
            independent = sum(abs(r['actual'] - r[column]) for r in records) / len(records)
            self.assertAlmostEqual(report[key], independent)
        for record in records:
            self.assertEqual(record['actual'], self.rows[record['row_index']]['Effort-'])
        self.assertEqual(report['split']['evaluation_samples'], len(records))
        self.assertEqual(report['split']['training_samples'] + len(records), len(self.rows))

    def test_heldout_targets_cannot_affect_fitting_or_predictions(self):
        changed = copy.deepcopy(self.rows)
        for i in self.report['split']['evaluation_indices']:
            changed[i]['Effort-'] = changed[i]['Effort-'] * 10000 + 999
        model, report = evaluate(changed)
        self.assertEqual(report['split'], self.report['split'])
        self.assertEqual(report['baseline'], self.report['baseline'])
        self.assertNotEqual(report['model_mae'], self.report['model_mae'])
        self.assertEqual([r['predicted'] for r in report['held_out_predictions']],
                         [r['predicted'] for r in self.report['held_out_predictions']])
        for field in ['missing', 'categorical']:
            left = model.named_steps[field]
            right = self.model.named_steps[field]
            if field == 'missing':
                np.testing.assert_array_equal(left.statistics_, right.statistics_)
            else:
                for a, b in zip(left.categories_, right.categories_):
                    np.testing.assert_array_equal(a, b)
        for left, right in zip(model.named_steps['forest'].estimators_, self.model.named_steps['forest'].estimators_):
            np.testing.assert_array_equal(left.tree_.value, right.tree_.value)
            np.testing.assert_array_equal(left.tree_.threshold, right.tree_.threshold)
        self.assertEqual(estimate(model, {}), estimate(self.model, {}))

    def test_preprocessing_sees_training_only(self):
        rows = fixture()
        train, heldout, _ = split_projects(rows)
        with patch('predict_effort.fit_training', wraps=fit_training) as spy:
            model, _ = evaluate(rows)
        self.assertEqual(spy.call_args.args[0], [rows[i] for i in train])
        learned = set(model.named_steps['categorical'].categories_[0])
        self.assertEqual(learned, {rows[i]['prec'] for i in train})
        self.assertTrue(learned.isdisjoint({rows[i]['prec'] for i in heldout}))
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update({'Kloc+': 999999, 'Defects-': 88888, 'idX': 'changed', 'MonthsX': 888})
        _, report = evaluate(changed)
        self.assertEqual(report, evaluate(rows)[1])

    def test_insufficient_groups_invalid_targets(self):
        for rows in [[], [dict.fromkeys(PREDICTORS, 'n') | {'Effort-': 2}] * 4]:
            with self.assertRaises(ValueError):
                evaluate(rows)
        for value in [-1, float('inf'), float('nan'), 'bad']:
            rows = fixture()
            rows[0]['Effort-'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate(rows)
        with self.assertRaises(ValueError):
            fit_training([])

    def test_json_and_cli_reject_forbidden_fields(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            proposal = Path(directory) / 'proposal.json'
            for text in ['{"prec":"n","prec":"h"}', '{bad', '[]', '{"Effort-":1}']:
                proposal.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_proposal(proposal)
            proposal.write_text('{"Kloc+":10}')
            result = subprocess.run([sys.executable, str(ROOT / 'source/predict_effort.py'), str(DATASET),
                                     '--proposal', str(proposal), '--evaluation', str(Path(directory) / 'out.json')],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Forbidden or unknown', result.stderr)
            self.assertFalse((Path(directory) / 'out.json').exists())
            proposal.write_text('{"prec": " n "}')
            self.assertEqual(load_proposal(proposal), {'prec': ' n '})


if __name__ == '__main__':
    unittest.main(verbosity=2)
