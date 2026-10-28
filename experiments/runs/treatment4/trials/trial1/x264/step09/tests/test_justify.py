"""Independent examples and full-dataset checks for retrospective justification."""

import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
from remember import Measurements, OBJECTIVE, load_measurements, summarize
from blame import diagnose
import justify


class JustifyTests(unittest.TestCase):
    def example(self):
        # y=6,0,2,0; median=1, mean=2. x=0 mean=3, x=1 mean=1.
        return Measurements(('x', 'constant'),
                            ((0, 9, 6), (0, 9, 0), (1, 9, 2), (1, 9, 0)))

    def test_selection_exact_ties_and_historical_reference(self):
        r = justify.justify(self.example())
        self.assertEqual(r['recommendation'], {'data_row': 2, 'settings': {'x': 0, 'constant': 9}, OBJECTIVE: 0})
        self.assertEqual(r['selection']['tied_best_data_rows'], [2, 4])
        self.assertEqual(r['selection']['tied_best_count'], 2)
        ref = r['historical_reference']
        self.assertEqual(ref[OBJECTIVE], 1)
        self.assertEqual(ref['difference_recommended_minus_reference'], -1)
        self.assertEqual(ref['observed_reduction_reference_minus_recommended'], 1)
        self.assertEqual(ref['observed_reduction_percent_of_reference'], 100)
        d = r['historical_distribution']
        self.assertEqual([d[k] for k in ('strictly_better_count', 'equal_count', 'strictly_worse_count')], [0, 2, 2])
        self.assertEqual(d['strictly_worse_percent'], 50)
        self.assertEqual(d['empirical_cdf_at_recommendation'], .5)
        self.assertEqual(d['statistics']['q25'], 0)
        self.assertEqual(d['statistics']['q75'], 3)
        self.assertEqual(d['competition_rank'], 1)

    def test_group_association_does_not_explain_individual_minimum(self):
        r = justify.justify(self.example())
        x, c = r['setting_evidence']
        # The chosen minimum uses a value whose marginal group mean is HIGHER.
        self.assertEqual(x['recommended_value'], 0)
        self.assertEqual(x['selected_value_group']['objective']['mean'], 3)
        self.assertEqual(x['selected_value_group']['mean_minus_overall'], 1)
        self.assertAlmostEqual(x['eta_squared'], 1 / 6)
        self.assertIsNone(c['eta_squared'])
        self.assertEqual(c['selected_value_group']['count'], 4)
        self.assertIn('not an individual contribution', x['interpretation'])

    def test_same_configuration_outcomes_not_replication_claim(self):
        r = justify.justify(self.example())
        self.assertEqual(r['selection']['same_configuration_data_rows'], [1, 2])
        self.assertEqual(r['selection']['same_configuration_observed_values'], [6, 0])
        self.assertEqual(r['historical_distribution']['repeated_configuration_rows'], 2)

    def test_near_ties_and_order_are_explicit(self):
        data = Measurements(('x',), ((9, 1 + 1e-12), (8, 1), (7, 1)))
        r = justify.justify(data)
        self.assertEqual(r['recommendation']['data_row'], 2)
        self.assertEqual(r['selection']['tied_best_data_rows'], [2, 3])
        reverse = justify.justify(Measurements(data.settings, tuple(reversed(data.rows))))
        self.assertEqual(reverse['recommendation']['settings'], {'x': 7})

    def test_single_constant_all_tied_zero_and_negative_references(self):
        for values in ((7,), (4, 4, 4), (0, 0), (-4, -2, 0)):
            with self.subTest(values=values):
                r = justify.justify(Measurements(('x',), tuple((0, y) for y in values)))
                ref = r['historical_reference']
                self.assertEqual(ref[OBJECTIVE], statistics.median(values))
                self.assertEqual(r['recommendation']['data_row'], 1)
                if ref[OBJECTIVE] <= 0:
                    self.assertIsNone(ref['observed_reduction_percent_of_reference'])
                self.assertEqual(r['setting_evidence'][0]['interpretation'],
                                 'Constant setting; no between-value comparison.')
                json.dumps(r, allow_nan=False)

    def test_preserved_diagnosis_and_summary_consistency(self):
        data = self.example()
        r = justify.justify(data)
        self.assertEqual(r['diagnosis'], diagnose(data))
        self.assertEqual(r['recommendation'], summarize(data)['top_runs'][0])
        for e in r['setting_evidence']:
            f = next(f for f in r['diagnosis']['ranked_findings'] + r['diagnosis']['constant_settings']
                     if f['setting'] == e['setting'])
            self.assertEqual(e['all_value_groups'], f['values'])
            self.assertEqual(e['eta_squared'], f['score'])
            self.assertIn(e['selected_value_group'], f['values'])
        self.assertIsNone(r['model_predictions'])
        self.assertEqual(r, justify.justify(data))

    def test_invalid_direct_inputs(self):
        bad = [None, {}, Measurements((), ((1,),)), Measurements(('x',), ()),
               Measurements(('x', 'x'), ((1, 2, 3),)),
               Measurements(('Performance-',), ((1, 2),)),
               Measurements((' ',), ((1, 2),)), Measurements(('x',), ((1,),))]
        for v in (True, '1', None, float('nan'), float('inf'), 10 ** 400, 1j):
            bad.extend([Measurements(('x',), ((v, 2),)), Measurements(('x',), ((0, v),))])
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                justify.justify(data)

    def test_actual_dataset_independent_evidence(self):
        path = Path(os.environ['X264_DATASET'])
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        y = [Fraction(row[OBJECTIVE]) for row in rows]
        best = min(y)
        index = y.index(best)
        r = justify.justify(load_measurements(path))
        self.assertEqual(r['recommendation']['data_row'], index + 1)
        self.assertEqual(r['recommendation'][OBJECTIVE], float(best))
        self.assertEqual(r['recommendation']['settings'],
                         {k: float(v) for k, v in rows[index].items() if k != OBJECTIVE})
        self.assertEqual(r['selection']['tied_best_data_rows'], [i + 1 for i, v in enumerate(y) if v == best])
        ref = statistics.median(y)
        self.assertAlmostEqual(r['historical_reference'][OBJECTIVE], float(ref))
        self.assertAlmostEqual(r['historical_reference']['difference_recommended_minus_reference'], float(best - ref))
        self.assertAlmostEqual(r['historical_reference']['observed_reduction_percent_of_reference'], float((ref - best) / ref * 100))
        self.assertEqual(r['historical_distribution']['strictly_worse_count'], sum(v > best for v in y))
        for e in r['setting_evidence']:
            group_y = [value for row, value in zip(rows, y)
                       if float(row[e['setting']]) == e['recommended_value']]
            g = e['selected_value_group']
            self.assertEqual(g['count'], len(group_y))
            self.assertAlmostEqual(g['objective']['mean'], float(sum(group_y) / len(group_y)))

    def test_cli_outputs_provenance_reproducibility_and_input_preservation(self):
        path = Path(os.environ['X264_DATASET'])
        before = path.read_bytes()
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            args = [sys.executable, '-B', str(STEP / 'source/justify.py'), str(path), '--output-dir', tmp]
            p = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            report_path = Path(tmp) / 'justification.json'
            first = report_path.read_bytes()
            r = json.loads(first)
            self.assertEqual(r['provenance']['sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(p.stdout, (Path(tmp) / 'justification.txt').read_text())
            self.assertEqual(p.stdout, justify.render_text(r))
            again = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(first, report_path.read_bytes())
        self.assertEqual(path.read_bytes(), before)

    def test_cli_invalid_missing_and_overwrite_inputs(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            tmp = Path(tmp)
            for name in ('justification.json', 'justification.txt'):
                path = tmp / name
                path.write_text('x,Performance-\n0,2\n1,3\n')
                before = path.read_bytes()
                with self.assertRaises(SystemExit) as caught:
                    justify.main([str(path), '--output-dir', str(tmp)])
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(path.read_bytes(), before)
            for contents in ('x,Performance-\n0,nan\n', 'x,Performance-\n',
                             'x,x,Performance-\n0,0,2\n', 'x,Performance-\n0\n'):
                path = tmp / 'bad.csv'
                path.write_text(contents)
                p = subprocess.run([sys.executable, '-B', str(STEP / 'source/justify.py'),
                                    str(path), '--output-dir', str(tmp / 'out')], capture_output=True)
                self.assertEqual(p.returncode, 2)
                self.assertFalse((tmp / 'out').exists())
            p = subprocess.run([sys.executable, '-B', str(STEP / 'source/justify.py'),
                                str(tmp / 'missing.csv'), '--output-dir', str(tmp / 'out')], capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse((tmp / 'out').exists())


if __name__ == '__main__':
    unittest.main()
