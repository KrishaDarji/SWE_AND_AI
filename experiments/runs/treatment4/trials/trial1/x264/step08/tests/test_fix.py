"""Independent exhaustive checks and edge cases for recorded-row repair."""

import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
from remember import Measurements, OBJECTIVE, load_measurements
import fix


class RepairTests(unittest.TestCase):
    def test_distance_before_objective_and_complete_changes(self):
        data = Measurements(('a', 'b', 'constant'),
                            ((0, 0, 9, 10), (1, 1, 9, 1), (1, 0, 9, 8)))
        r = fix.recommend_repair(data, 1)
        self.assertEqual(r['recommendation']['data_row'], 3)
        self.assertEqual(r['change_cost'], 1)
        self.assertEqual(r['changed_columns'], [{'column': 'a', 'old': 0, 'new': 1}])
        self.assertEqual(r['observed_objective_reduction'], 2)
        self.assertEqual(r['baseline']['settings'], {'a': 0, 'b': 0, 'constant': 9})

    def test_objective_then_row_ties(self):
        data = Measurements(('x',), ((0, 10), (1, 8), (2, 6), (3, 6)))
        self.assertEqual(fix.recommend_repair(data, 1)['recommendation']['data_row'], 3)
        reversed_data = Measurements(('x',), ((0, 10), (3, 6), (2, 6), (1, 8)))
        self.assertEqual(fix.recommend_repair(reversed_data, 1)['recommendation']['settings'], {'x': 3})

    def test_strict_improvement_and_no_tolerance(self):
        data = Measurements(('x',), ((0, 1), (1, 1), (2, 1 - 1e-12)))
        self.assertEqual(fix.recommend_repair(data, 1)['recommendation']['data_row'], 3)
        self.assertEqual(fix.recommend_repair(data, 3)['status'], 'no_improving_candidate')

    def test_no_improvement_singleton_ties_and_global_minimum(self):
        for rows in (((0, 4),), ((0, 4), (1, 4)), ((0, -3), (1, 0))):
            with self.subTest(rows=rows):
                r = fix.recommend_repair(Measurements(('x',), rows), 1)
                self.assertEqual(r['status'], 'no_improving_candidate')
                for key in ('recommendation', 'change_cost', 'observed_objective_reduction'):
                    self.assertIsNone(r[key])
                self.assertEqual(r['changed_columns'], [])
                self.assertIn('No improving candidate', fix.render_text(r))

    def test_duplicate_configurations_zero_cost_and_duplicate_rows(self):
        data = Measurements(('x',), ((0, 10), (0, 8), (0, 8), (1, 1)))
        r = fix.recommend_repair(data, 1)
        self.assertEqual(r['recommendation']['data_row'], 2)
        self.assertEqual(r['change_cost'], 0)
        self.assertEqual(r['changed_columns'], [])
        self.assertEqual(r['observed_objective_reduction'], 2)
        self.assertEqual(r['search']['improving_row_count'], 3)
        self.assertIn('identical recorded settings', fix.render_text(r))

    def test_every_changed_column_in_original_order(self):
        r = fix.recommend_repair(Measurements(('z', 'a', 'b'),
                                              ((0, 1, 2, 10), (4, 3, 5, 0))), 1)
        self.assertEqual(r['changed_columns'], [
            {'column': 'z', 'old': 0, 'new': 4},
            {'column': 'a', 'old': 1, 'new': 3},
            {'column': 'b', 'old': 2, 'new': 5}])
        self.assertEqual(r['change_cost'], 3)

    def test_invalid_baseline_rows(self):
        data = Measurements(('x',), ((0, 10), (1, 5)))
        for row in (0, -1, 3, True, False, 1.0, '1', None):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'baseline_row'):
                fix.recommend_repair(data, row)
        self.assertEqual(fix.recommend_repair(data, 2)['baseline']['data_row'], 2)

    def test_invalid_direct_data_and_reduction_overflow(self):
        for data in (None, Measurements((), ((1,),)), Measurements(('x',), ()),
                     Measurements(('x',), ((0, float('nan')),)),
                     Measurements(('x',), ((True, 1),))):
            with self.subTest(data=data), self.assertRaises(ValueError):
                fix.recommend_repair(data, 1)
        with self.assertRaisesRegex(ValueError, 'finite numeric range'):
            fix.recommend_repair(Measurements(('x',), ((0, 1e308), (1, -1e308))), 1)

    def test_randomized_independent_exhaustive_oracle(self):
        rng = random.Random(8031)
        for _ in range(20):
            rows = tuple(tuple(rng.randrange(3) for _ in range(4)) + (rng.randrange(-5, 10),)
                         for _ in range(20))
            data = Measurements(('a', 'b', 'c', 'd'), rows)
            for baseline in range(1, len(rows) + 1):
                r = fix.recommend_repair(data, baseline)
                candidates = [i for i in range(len(rows)) if rows[i][-1] < rows[baseline-1][-1]]
                if not candidates:
                    self.assertIsNone(r['recommendation'])
                    continue
                # Independent materialized exhaustive ordering using changed-index sets.
                distances = {i: len({j for j in range(4)
                                     if rows[i][j] != rows[baseline-1][j]}) for i in candidates}
                ordered = sorted(candidates, key=lambda i: (distances[i], rows[i][-1], i))
                self.assertEqual(r['recommendation']['data_row'], ordered[0] + 1)
                self.assertEqual(r['change_cost'], distances[ordered[0]])
                self.assertEqual(r['change_cost'], len(r['changed_columns']))
                self.assertGreater(r['observed_objective_reduction'], 0)

    def test_dataset_independent_fraction_exhaustive_search(self):
        path = Path(os.environ['X264_DATASET'])
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns = [c for c in reader.fieldnames if c != OBJECTIVE]
            rows = [{k: Fraction(v) for k, v in row.items()} for row in reader]
        baseline = rows[0]
        candidates = [(sum(row[c] != baseline[c] for c in columns), row[OBJECTIVE], i)
                      for i, row in enumerate(rows, 1) if row[OBJECTIVE] < baseline[OBJECTIVE]]
        cost, value, number = sorted(candidates)[0]
        data = load_measurements(path)
        r = fix.recommend_repair(data, 1)
        self.assertEqual(r['recommendation']['data_row'], number)
        self.assertEqual(r['recommendation'][OBJECTIVE], float(value))
        self.assertEqual(r['change_cost'], cost)
        self.assertAlmostEqual(r['observed_objective_reduction'], float(baseline[OBJECTIVE] - value))
        self.assertEqual(r['changed_columns'], [
            {'column': c, 'old': float(baseline[c]), 'new': float(rows[number-1][c])}
            for c in columns if baseline[c] != rows[number-1][c]])
        self.assertEqual(r, fix.recommend_repair(data, 1))
        minimum_row = min(range(len(rows)), key=lambda i: rows[i][OBJECTIVE]) + 1
        self.assertEqual(fix.recommend_repair(data, minimum_row)['status'], 'no_improving_candidate')

    def test_cli_reproducibility_provenance_and_input_preservation(self):
        path = Path(os.environ['X264_DATASET'])
        before = path.read_bytes()
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            args = [sys.executable, '-B', str(STEP / 'source/fix.py'), str(path),
                    '--baseline-row', '1', '--output-dir', tmp]
            first = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            artifact = Path(tmp) / 'repair.json'
            saved = artifact.read_bytes()
            r = json.loads(saved)
            self.assertEqual(r['provenance']['sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(first.stdout, (Path(tmp) / 'repair.txt').read_text())
            self.assertEqual(first.stdout, fix.render_text(r))
            second = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(saved, artifact.read_bytes())
        self.assertEqual(before, path.read_bytes())

    def test_cli_bad_baselines_missing_malformed_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            tmp = Path(tmp)
            dataset = tmp / 'data.csv'
            dataset.write_text('x,Performance-\n0,2\n1,1\n')
            for row in ('0', '-1', '3', '1.5', 'true'):
                p = subprocess.run([sys.executable, '-B', str(STEP / 'source/fix.py'),
                                    str(dataset), '--baseline-row', row, '--output-dir', str(tmp / 'out')],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertFalse((tmp / 'out').exists())
            for path in (tmp / 'missing.csv', tmp / 'bad.csv'):
                if path.name == 'bad.csv':
                    path.write_text('x,Performance-\n0,nan\n')
                p = subprocess.run([sys.executable, '-B', str(STEP / 'source/fix.py'),
                                    str(path), '--baseline-row', '1', '--output-dir', str(tmp / 'out')],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
            for name in ('repair.json', 'repair.txt'):
                path = tmp / name
                path.write_bytes(dataset.read_bytes())
                before = path.read_bytes()
                with self.assertRaises(SystemExit) as error:
                    fix.main([str(path), '--baseline-row', '1', '--output-dir', str(tmp)])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
