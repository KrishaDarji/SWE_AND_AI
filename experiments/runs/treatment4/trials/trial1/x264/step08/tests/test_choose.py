"""Budget, information-boundary, model-training, and post-freeze audit tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
import choose
from remember import OBJECTIVE


class ChooseTests(unittest.TestCase):
    def candidates(self, n=16):
        return tuple(choose.Candidate(i + 1, (float(i % 4), float(i // 4))) for i in range(n))

    def run_selection(self, *, budget=12, seed=42, values=None):
        candidates = self.candidates()
        values = values if values is not None else [float((i * 7) % 13) for i in range(16)]
        calls = []

        def evaluate(row):
            calls.append(row)
            if len(calls) > budget:
                self.fail('Budget exceeded')
            return values[row - 1]

        report = choose.select(('a', 'b'), candidates, evaluate, budget=budget, seed=seed)
        return report, calls

    def test_budget_enforcement_and_best_observed(self):
        for budget in (1, 3, 5, 6, 12, 16):
            report, calls = self.run_selection(budget=budget)
            history = report['evaluation_history']
            self.assertEqual(report['actual_evaluation_count'], budget)
            self.assertEqual(len(calls), budget)
            self.assertEqual(len(set(calls)), budget)
            self.assertEqual(calls, [h['data_row'] for h in history])
            best = min(history, key=lambda h: h[OBJECTIVE])
            self.assertEqual(report['selected_candidate']['data_row'], best['data_row'])
            for end, event in enumerate(history, 1):
                prefix_best = min(history[:end], key=lambda h: h[OBJECTIVE])
                self.assertEqual(event['best_so_far']['data_row'], prefix_best['data_row'])
                self.assertEqual(event['best_so_far'][OBJECTIVE], prefix_best[OBJECTIVE])

    def test_reproducibility_and_seed_effect(self):
        self.assertEqual(self.run_selection(), self.run_selection())
        self.assertNotEqual(self.run_selection()[1][:5], self.run_selection(seed=43)[1][:5])

    def test_exact_ties_keep_first_evaluated_and_full_candidate_pool(self):
        report, calls = self.run_selection(budget=16, values=[3.] * 16)
        self.assertEqual(report['selected_candidate']['data_row'], calls[0])
        self.assertEqual(set(calls), set(range(1, 17)))
        # All surrogate scores tie, so acquisition uses the lowest remaining row.
        self.assertEqual(calls[5], min(set(range(1, 17)) - set(calls[:5])))

    def test_invalid_budgets_and_seeds_make_zero_calls(self):
        for budget in (0, -1, True, False, 1.5, '2', None, 17):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    choose.select(('a', 'b'), self.candidates(),
                                  lambda _: self.fail('Invalid budget evaluated'), budget=budget, seed=42)
        for seed in (-1, True, 2.5, None, 2 ** 32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                choose.select(('a', 'b'), self.candidates(),
                              lambda _: self.fail('Invalid seed evaluated'), budget=1, seed=seed)

    def test_invalid_candidate_schema_before_evaluation(self):
        cases = [(), (choose.Candidate(1, (1.,)),),
                 (choose.Candidate(1, (1., 2.)),) * 2,
                 (choose.Candidate(0, (1., 2.)),),
                 (choose.Candidate(1, (float('nan'), 2.)),)]
        for candidates in cases:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                choose.select(('a', 'b'), candidates, lambda _: self.fail('Invalid input evaluated'),
                              budget=1, seed=42)

    def test_surrogate_fits_only_counted_observations(self):
        real_model = choose.RandomForestRegressor
        calls, fits = [], []
        candidates = self.candidates()
        values = [float(i % 7) for i in range(16)]

        class CheckedForest(real_model):
            def fit(inner, x, y, *args, **kwargs):
                self.assertEqual(list(x), [candidates[row - 1].values for row in calls])
                self.assertEqual(list(y), [values[row - 1] for row in calls])
                fits.append(len(calls))
                return super().fit(x, y, *args, **kwargs)

        def evaluate(row):
            calls.append(row)
            return values[row - 1]

        with patch.object(choose, 'RandomForestRegressor', CheckedForest):
            choose.select(('a', 'b'), candidates, evaluate, budget=12, seed=42)
        self.assertEqual(fits, [5, 6, 7, 8, 10, 11])

    def test_selector_only_calls_objective_capability(self):
        calls = []

        class CallOnly:
            def __getattribute__(self, name):
                raise AssertionError(f'Selector inspected oracle attribute {name}')

            def __call__(self, row):
                calls.append(row)
                return float(row % 5)

        with patch('remember.load_measurements', side_effect=AssertionError('Bulk targets accessed')):
            report = choose.select(('a', 'b'), self.candidates(), CallOnly(), budget=12, seed=42)
        self.assertEqual(len(calls), report['actual_evaluation_count'])

    def test_unqueried_objectives_can_be_unparseable_without_affecting_selection(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            path = Path(directory) / 'data.csv'
            values = [float(i % 7) for i in range(16)]

            def write(hidden):
                path.write_text('a,b,Performance-\n' + ''.join(
                    f'{i % 4},{i // 4},{hidden.get(i + 1, values[i])}\n' for i in range(16)))

            write({})
            settings, candidates = choose.load_candidates(path)
            oracle = choose.CSVOracle(path, 12)
            first = choose.select(settings, candidates, oracle, budget=12, seed=42)
            queried = {h['data_row'] for h in first['evaluation_history']}
            write({row: 'SECRET_NOT_A_NUMBER' for row in range(1, 17) if row not in queried})
            next_settings, next_candidates = choose.load_candidates(path)
            self.assertEqual((settings, candidates), (next_settings, next_candidates))
            second_oracle = choose.CSVOracle(path, 12)
            second = choose.select(next_settings, next_candidates, second_oracle, budget=12, seed=42)
            self.assertEqual(first, second)
            self.assertEqual(second_oracle.count, 12)
            with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                second_oracle(next(iter(set(range(1, 17)) - queried)))
            self.assertEqual(second_oracle.count, 12)

    def test_oracle_charges_failed_parse_and_rejects_repeat(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('a,Performance-\n0,2\n1,bad\n')
            oracle = choose.CSVOracle(path, 2)
            self.assertEqual(oracle(1), 2)
            with self.assertRaisesRegex(ValueError, 'already evaluated'):
                oracle(1)
            self.assertEqual(oracle.count, 1)
            with self.assertRaises(ValueError):
                oracle(2)
            self.assertEqual(oracle.count, 2)
            with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                oracle(3)

    def test_separate_audit_does_not_modify_selection(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('Performance-,x\n10,0\n2,1\n')
            settings, candidates = choose.load_candidates(path)
            result = choose.select(settings, candidates, choose.CSVOracle(path, 1), budget=1, seed=42)
            frozen = json.dumps(result, sort_keys=True)
            audit = choose.post_selection_audit(path, frozen)
            self.assertEqual(json.dumps(result, sort_keys=True), frozen)
            self.assertEqual(audit['dataset_minimum'], 2)
            self.assertEqual(audit['absolute_gap'], 8)
            self.assertFalse(audit['fed_back_into_selection'])
            self.assertFalse(result['quality_constraint_verified'])
            self.assertIn('cannot be verified', result['quality_constraint_limitation'])

    def test_cli_freezes_selection_before_audit_and_preserves_dataset(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('x,Performance-\n0,10\n1,2\n')
            before = path.read_bytes()
            original_audit = choose.post_selection_audit

            def checked_audit(dataset, frozen):
                disk = json.loads((Path(directory) / 'selection.json').read_text())
                self.assertEqual(disk, json.loads(frozen))
                self.assertNotIn('post_selection_audit', disk)
                return original_audit(dataset, frozen)

            with patch.object(choose, 'post_selection_audit', side_effect=checked_audit):
                with patch('builtins.print'):
                    choose.main([str(path), '--output-dir', directory, '--budget', '1'])
            self.assertEqual(path.read_bytes(), before)
            result = json.loads((Path(directory) / 'selection.json').read_text())
            self.assertEqual(result['actual_evaluation_count'], 1)
            self.assertIn('post_selection_audit', result)

    def test_actual_dataset_cli_reproducibility_and_preservation(self):
        path = Path(os.environ['X264_DATASET'])
        before = path.read_bytes()
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            command = [sys.executable, '-B', str(STEP / 'source/choose.py'), str(path),
                       '--output-dir', directory, '--budget', '20', '--seed', '42']
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads((Path(directory) / 'selection.json').read_text())
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(first.stdout, again.stdout)
            self.assertEqual(report['actual_evaluation_count'], 20)
            self.assertEqual(report['candidate_count'], 1152)
            self.assertEqual(len(report['features']), 16)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_invalid_budget_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            path = Path(directory) / 'selection.json'
            path.write_text('x,Performance-\n0,2\n')
            before = path.read_bytes()
            for budget in ('0', '-1', '2'):
                with self.assertRaises(SystemExit) as caught:
                    choose.main([str(path), '--output-dir', str(Path(directory) / 'out'), '--budget', budget])
                self.assertEqual(caught.exception.code, 2)
            with self.assertRaises(SystemExit):
                choose.main([str(path), '--output-dir', directory, '--budget', '1'])
            self.assertEqual(before, path.read_bytes())
            self.assertFalse((Path(directory) / 'out').exists())


if __name__ == '__main__':
    unittest.main()
