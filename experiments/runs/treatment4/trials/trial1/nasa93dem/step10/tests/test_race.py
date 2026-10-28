"""Paired-repeat isolation, leakage, arithmetic, reproducibility and CLI checks."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from race import SEEDS, compare, outcome
from predict_effort import evaluate, fit_training, feature_matrix
from summarize import PREDICTORS, read_projects

DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


def fixture():
    rows = [dict.fromkeys(PREDICTORS, f'level{i}') | {'Effort-': float(i * 11)} for i in range(16)]
    rows.append(dict(rows[0], prec=' level0 ', **{'Effort-': 500.0}))
    rows.append(dict(rows[1], **{'Effort-': None}))
    return rows


def deterministic(report):
    report = copy.deepcopy(report)
    report['descriptive_summary'] = {k: v for k, v in report['descriptive_summary'].items() if 'runtime' not in k}
    for r in report['per_seed_results']:
        r.pop('model_runtime_seconds')
        r.pop('baseline_runtime_seconds')
    return report


class RaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_projects(DATASET)
        cls.report = compare(cls.rows)

    def test_all_seeds_partitions_arithmetic_and_counts(self):
        self.assertEqual(self.report['seeds'], SEEDS)
        for r in self.report['per_seed_results']:
            split = r['split']
            train, test = split['training_indices'], split['evaluation_indices']
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(train) | set(test), set(range(len(self.rows))))
            self.assertEqual(split['training_samples'], len(train))
            self.assertEqual(split['evaluation_samples'], len(test))
            self.assertFalse(set(split['training_group_ids']) & set(split['evaluation_group_ids']))
            self.assertEqual(split['evaluation_groups'], math.ceil(split['total_groups'] * .25))
            self.assertEqual(split['training_groups'] + split['evaluation_groups'], split['total_groups'])
            records = r['held_out_predictions']
            self.assertEqual([p['row_index'] for p in records], test)
            avg = sum(self.rows[i]['Effort-'] for i in train) / len(train)
            self.assertAlmostEqual(avg, r['baseline_training_mean'])
            for p in records:
                self.assertEqual(p['actual'], self.rows[p['row_index']]['Effort-'])
                self.assertAlmostEqual(p['baseline_predicted'], avg)
            for key, col in [('model_mae', 'predicted'), ('baseline_mae', 'baseline_predicted')]:
                self.assertAlmostEqual(r[key], sum(abs(p['actual'] - p[col]) for p in records) / len(records))
            self.assertEqual(r['paired_mae_difference'], r['model_mae'] - r['baseline_mae'])
            self.assertEqual(r['outcome'], outcome(r['model_mae'], r['baseline_mae']))
            self.assertGreaterEqual(r['model_runtime_seconds'], 0)
            self.assertGreaterEqual(r['baseline_runtime_seconds'], 0)
        counts = self.report['observed_outcomes']
        self.assertEqual(sum(counts.values()), 10)
        for key in counts:
            self.assertEqual(counts[key], sum(r['outcome'] == key for r in self.report['per_seed_results']))
        for key in ('model_mae', 'baseline_mae', 'paired_mae_difference'):
            self.assertAlmostEqual(self.report['descriptive_summary'][key]['mean'],
                                   sum(r[key] for r in self.report['per_seed_results']) / 10)

    def test_reproducibility_and_step2_preservation(self):
        self.assertEqual(deterministic(self.report), deterministic(compare(self.rows)))
        _, old = evaluate(self.rows)
        first = self.report['per_seed_results'][0]
        for key in ('training_indices', 'evaluation_indices'):
            self.assertEqual(first['split'][key], old['split'][key])
        for key in ('model_mae', 'baseline_mae', 'held_out_predictions'):
            self.assertEqual(first[key], old[key])
        self.assertEqual(self.report['model_settings'], old['model_settings'])
        self.assertEqual(self.report['statistical_analysis']['status'], 'pending')
        json.dumps(self.report, allow_nan=False)

    def test_normalized_groups_missing_targets_and_excluded_fields(self):
        rows = fixture()
        report = compare(rows, [42, 43])
        for r in report['per_seed_results']:
            s = r['split']
            self.assertEqual(s['excluded_missing_targets'], 1)
            self.assertEqual(s['eligible_samples'], 17)
            self.assertEqual(s['total_groups'], 16)
            for partition in ('training_indices', 'evaluation_indices'):
                self.assertEqual(0 in s[partition], 16 in s[partition])
                self.assertNotIn(17, s[partition])
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update({'Kloc+': 1e12, 'Defects-': -50, 'idX': 'x', 'futureX': 900})
        self.assertEqual(deterministic(report), deterministic(compare(changed, [42, 43])))

    def test_training_only_fresh_fits_and_target_leakage(self):
        rows = fixture()
        fits = []
        def capture(training):
            model, baseline = fit_training(training)
            fits.append((copy.deepcopy(training), model))
            return model, baseline
        with patch('race.fit_training', side_effect=capture):
            report = compare(rows, [42, 43])
        self.assertIsNot(fits[0][1], fits[1][1])
        for result, (training, model) in zip(report['per_seed_results'], fits):
            self.assertEqual(training, [rows[i] for i in result['split']['training_indices']])
            self.assertEqual(set(model.named_steps['categorical'].categories_[0]), {r['prec'].strip() for r in training})
            altered = copy.deepcopy(rows)
            for i in result['split']['evaluation_indices']:
                altered[i]['Effort-'] = 1e9 + i
            with patch('race.fit_training', wraps=fit_training) as spy:
                repeat = compare(altered, [result['seed']])['per_seed_results'][0]
            self.assertEqual(spy.call_args.args[0], training)
            self.assertEqual(result['split'], repeat['split'])
            self.assertEqual(result['baseline_training_mean'], repeat['baseline_training_mean'])
            self.assertEqual([r['predicted'] for r in result['held_out_predictions']],
                             [r['predicted'] for r in repeat['held_out_predictions']])
            self.assertNotEqual(result['model_mae'], repeat['model_mae'])

    def test_ties_and_invalid_inputs(self):
        self.assertEqual(outcome(1, 1), 'tie')
        self.assertEqual(outcome(0, 1e-12), 'tie')
        self.assertEqual(outcome(0, 2e-12), 'win')
        self.assertEqual(outcome(2, 1), 'loss')
        for value in (-1, True, 'bad', float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                outcome(value, 0)
        constant = fixture()
        for row in constant:
            row['Effort-'] = 5.0
        self.assertEqual(compare(constant, [42])['observed_outcomes'], {'win': 0, 'tie': 1, 'loss': 0})
        for seeds in ([], [42, 42], [True], [-1], [2**32], ['42'], [1.2]):
            with self.assertRaises(ValueError):
                compare(fixture(), seeds)
        for rows in ([], None, [1], [{'Effort-': None}], [{'Effort-': 1}] * 4):
            with self.assertRaises(ValueError):
                compare(rows, [42])
        for value in (True, -1, float('inf'), float('nan'), 'bad'):
            rows = fixture()
            rows[0]['Effort-'] = value
            with self.assertRaises(ValueError):
                compare(rows, [42])
        rows = fixture()
        rows[0]['prec'] = 3
        with self.assertRaises(ValueError):
            compare(rows, [42])

    def test_cli_arguments(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            output = Path(tmp) / 'race.json'
            command = [sys.executable, str(ROOT / 'source/race.py'), str(DATASET), '--output', str(output), '--seeds']
            good = subprocess.run(command + ['44', '45'], capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(output.read_text())['seeds'], [44, 45])
            bad = subprocess.run(command + ['42', '42'], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn('Seeds', bad.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
