"""Independently checkable association scores, directions and integration."""

import csv
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
import blame
from remember import Measurements, load_measurements


class BlameTests(unittest.TestCase):
    def test_exact_direction_score_and_constant(self):
        # y = 0,2,4,6; mean=3; SST=20. a groups mean 1,5: SSB=16.
        # b groups mean 2,4: SSB=4. Scores .8 and .2; constant excluded.
        data = Measurements(('a', 'b', 'constant'),
                            ((0, 0, 9, 0), (0, 1, 9, 2),
                             (1, 0, 9, 4), (1, 1, 9, 6)))
        r = blame.diagnose(data)
        a, b = r['ranked_findings']
        self.assertEqual([a['setting'], b['setting']], ['a', 'b'])
        self.assertAlmostEqual(a['score'], .8)
        self.assertAlmostEqual(b['score'], .2)
        self.assertEqual(a['highest_mean_values'], [1])
        self.assertEqual(a['lowest_mean_values'], [0])
        self.assertEqual([v['count'] for v in a['values']], [2, 2])
        self.assertEqual([v['mean_minus_overall'] for v in a['values']], [-2, 2])
        self.assertEqual([v['association'] for v in a['values']], ['lower', 'higher'])
        c = r['constant_settings'][0]
        self.assertEqual(c['setting'], 'constant')
        self.assertIsNone(c['score'])
        self.assertIsNone(c['rank'])
        reverse = blame.diagnose(Measurements(data.settings, tuple(
            (*row[:-1], -row[-1]) for row in data.rows)))
        self.assertEqual(reverse['ranked_findings'][0]['score'], a['score'])
        self.assertEqual(reverse['ranked_findings'][0]['highest_mean_values'], [0])

    def test_ties_redundancy_and_repeatability(self):
        data = Measurements(('z', 'a', 'other'),
                            ((0, 1, 0, 0), (0, 1, 1, 2),
                             (1, 0, 0, 4), (1, 0, 1, 6)))
        r = blame.diagnose(data)
        self.assertEqual(r, blame.diagnose(data))
        self.assertEqual([f['setting'] for f in r['ranked_findings']], ['a', 'z', 'other'])
        self.assertEqual([f['rank'] for f in r['ranked_findings']], [1, 1, 3])
        pair = next(p for p in r['setting_relationships'] if set(p['settings']) == {'a', 'z'})
        self.assertAlmostEqual(pair['pearson_r'], -1)
        self.assertTrue(pair['equivalent_observed_partitions'])
        self.assertIsNone(r['method']['seed'])
        self.assertIsNone(r['method']['fitted_model'])

    def test_multilevel_unequal_counts(self):
        # y=0,2,4,8; overall=3.5. SST=35; SSB=33, within SS=2.
        data = Measurements(('level',), ((0, 0), (0, 2), (1, 4), (2, 8)))
        f = blame.diagnose(data)['ranked_findings'][0]
        self.assertAlmostEqual(f['score'], 33 / 35)
        self.assertEqual([g['count'] for g in f['values']], [2, 1, 1])
        self.assertEqual(f['highest_mean_values'], [2])
        self.assertEqual(f['lowest_mean_values'], [0])

    def test_zero_variance_single_row_and_equal_means(self):
        r = blame.diagnose(Measurements(('x',), ((0, 7), (1, 7))))
        f = r['ranked_findings'][0]
        self.assertEqual(f['score'], 0)
        self.assertEqual(f['highest_mean_values'], [0, 1])
        self.assertIn('no directional', f['explanation'])
        r = blame.diagnose(Measurements(('x',), ((0, 7),)))
        self.assertEqual(r['ranked_findings'], [])
        self.assertEqual(len(r['constant_settings']), 1)
        r = blame.diagnose(Measurements(('x',), ((0, 0), (0, 4), (1, 1), (1, 3))))
        self.assertEqual(r['ranked_findings'][0]['score'], 0)

    def test_invalid_direct_inputs(self):
        bad = [None, Measurements((), ((1,),)), Measurements(('x',), ()),
               Measurements(('Performance-',), ((1, 2),)),
               Measurements(('x', 'x'), ((1, 2, 3),)),
               Measurements((' ',), ((1, 2),)), Measurements(('x',), ((1,),)),
               Measurements(('x',), (None,))]
        for v in (True, '1', None, float('nan'), float('inf'), 10 ** 400, 1j):
            bad += [Measurements(('x',), ((v, 2),)), Measurements(('x',), ((1, v),))]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                blame.diagnose(data)

    def test_actual_scores_match_independent_fraction_calculation(self):
        path = Path(os.environ['X264_DATASET'])
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        r = blame.diagnose(load_measurements(path))
        y = [Fraction(row['Performance-']) for row in rows]
        mean = sum(y) / len(y)
        ss = sum((v - mean) ** 2 for v in y)
        expected = {}
        for name in rows[0]:
            if name == 'Performance-':
                continue
            groups = {}
            for row, value in zip(rows, y):
                groups.setdefault(row[name], []).append(value)
            if len(groups) == 1:
                continue
            expected[name] = float(sum(len(g) * (sum(g) / len(g) - mean) ** 2
                                       for g in groups.values()) / ss)
        self.assertEqual([f['setting'] for f in r['ranked_findings']],
                         sorted(expected, key=lambda n: (-expected[n], n)))
        for f in r['ranked_findings']:
            self.assertAlmostEqual(f['score'], expected[f['setting']], places=12)
            self.assertEqual(sum(g['count'] for g in f['values']), len(rows))
        self.assertEqual({f['setting'] for f in r['constant_settings']},
                         {'no_asm', 'rc_lookahead', 'ref'})
        self.assertNotIn('Performance-', r['features'])

    def test_cli_outputs_preserves_dataset_and_is_repeatable(self):
        path = Path(os.environ['X264_DATASET'])
        before = path.read_bytes()
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            args = [sys.executable, '-B', str(STEP / 'source/blame.py'), str(path),
                    '--output-dir', tmp]
            process = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = Path(tmp) / 'blame.json'
            contents = report.read_bytes()
            self.assertEqual((Path(tmp) / 'blame.txt').read_text(), process.stdout)
            result = json.loads(contents)
            self.assertEqual(result['row_count'], 1152)
            again = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(contents, report.read_bytes())
        self.assertEqual(before, path.read_bytes())

    def test_cli_invalid_csv_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            path = Path(tmp) / 'blame.json'
            path.write_text('x,Performance-\n0,2\n1,3\n')
            before = path.read_bytes()
            with self.assertRaises(SystemExit) as result:
                blame.main([str(path), '--output-dir', tmp])
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(before, path.read_bytes())
            path.write_text('x,Performance-\n0,nan\n')
            process = subprocess.run([sys.executable, '-B', str(STEP / 'source/blame.py'),
                                      str(path), '--output-dir', str(Path(tmp) / 'output')],
                                     text=True, capture_output=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn('nonfinite', process.stderr)
            self.assertFalse((Path(tmp) / 'output').exists())


if __name__ == '__main__':
    unittest.main()
