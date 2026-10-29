"""Analytic statistical fixtures, fixed membership, missingness, and actual CSV checks."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from certify import compare_families, select_families, welch_comparison
from summarize import CONFIGURATION, load_rows

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def fixture():
    families = ['squared_error'] * 3 + ['absolute_error'] * 3 + ['poisson', None]
    return [dict(zip(CONFIGURATION, [i, family, 2, 0, 5]),
                 **{'MRE-': value, 'ACC+': value + 1, 'PRED40+': value + 2})
            for i, (family, value) in enumerate(zip(families, [3, 4, 5, 1, 2, 3, 90, 99]))]


class CertifyTests(unittest.TestCase):
    def test_hand_calculated_welch(self):
        # Means 4 and 2, sample variances 1 each, SE sqrt(2/3), df=4.
        # For df=4, two-sided t tail at sqrt(6) is 1-3u/2+u^3/2, u=sqrt(.6).
        r = welch_comparison([3, 4, 5], [1, 2, 3])
        self.assertEqual(r['candidate_minus_baseline_mean'], -2)
        self.assertAlmostEqual(r['standard_error'], math.sqrt(2 / 3))
        self.assertAlmostEqual(r['degrees_of_freedom'], 4)
        self.assertAlmostEqual(r['t_statistic'], -math.sqrt(6))
        u = math.sqrt(.6)
        self.assertAlmostEqual(r['p_two_sided'], 1 - 1.5 * u + .5 * u ** 3)
        margin = 2.7764451051977987 * math.sqrt(2 / 3)
        for actual, expected in zip(r['confidence_interval'], [-2 - margin, -2 + margin]):
            self.assertAlmostEqual(actual, expected)
        self.assertFalse(r['statistically_detected'])
        self.assertEqual(r['observed_direction'], 'candidate_better')

    def test_improvement_direction_and_unequal_variance(self):
        r = welch_comparison([10, 11, 12], [0, 2, 4, 6])
        # Variance contributions: 1/3 and (20/3)/4 = 5/3.
        self.assertEqual(r['candidate_minus_baseline_mean'], -8)
        self.assertAlmostEqual(r['standard_error'], math.sqrt(2))
        self.assertAlmostEqual(r['degrees_of_freedom'], 4 / ((1 / 3) ** 2 / 2 + (5 / 3) ** 2 / 3))
        self.assertTrue(r['statistically_detected'])
        self.assertEqual(r['observed_direction'], 'candidate_better')
        self.assertEqual(welch_comparison([10, 11, 12], [0, 2, 4, 6], 'maximize')['observed_direction'], 'candidate_worse')
        self.assertEqual(welch_comparison([0, 2, 4, 6], [10, 11, 12])['observed_direction'], 'candidate_worse')

    def test_identical_samples_and_degenerate_constants(self):
        r = welch_comparison([1, 2, 3], [1, 2, 3])
        self.assertEqual(r['p_two_sided'], 1)
        self.assertEqual(r['candidate_minus_baseline_mean'], 0)
        self.assertEqual(r['observed_direction'], 'equal')
        self.assertFalse(r['statistically_detected'])
        for candidate in ([1, 1], [2, 2]):
            r = welch_comparison([1, 1], candidate)
            self.assertEqual(r['status'], 'zero_variance_inference_unavailable')
            self.assertIsNone(r['p_two_sided'])
        self.assertEqual(welch_comparison([1, 1], [1, 2])['status'], 'ok')

    def test_small_and_invalid_samples(self):
        for a, b in [([], []), ([1], [2, 3]), ([None], [2, 3]), ([1, 2], [])]:
            r = welch_comparison(a, b)
            self.assertEqual(r['status'], 'insufficient_data')
            self.assertIsNone(r['confidence_interval'])
        for value in [float('nan'), float('inf'), '1', True]:
            with self.assertRaises(ValueError):
                welch_comparison([value, 2], [1, 2])
        for threshold in [0, -1, math.inf, math.nan, True, '1']:
            with self.assertRaises(ValueError):
                welch_comparison([1, 2], [3, 4], practical_threshold=threshold)
        with self.assertRaises(ValueError):
            welch_comparison([1, 2], [3, 4], 'invalid')
        with self.assertRaises(ValueError):
            compare_families(fixture(), primary='unknown')
        for field, a, b in [('MRE-', 1, 2), ('criterion', 'a', 'a'), ('criterion', None, 'b')]:
            with self.assertRaises(ValueError):
                select_families(fixture(), field, a, b)

    def test_missing_outcomes_exclusions_and_no_mutation(self):
        rows = fixture()
        rows[0]['MRE-'] = None
        rows[3]['ACC+'] = None
        saved = copy.deepcopy(rows)
        r = compare_families(rows)
        self.assertEqual(rows, saved)
        self.assertEqual(r['excluded_count'], 2)
        self.assertEqual(r['excluded_missing_family_count'], 1)
        self.assertEqual(r['included_count'], 6)
        self.assertEqual(r['groups']['baseline']['objectives']['MRE-']['missing'], 1)
        self.assertEqual(r['groups']['baseline']['objectives']['MRE-']['n'], 2)
        self.assertEqual(r['groups']['candidate']['objectives']['ACC+']['missing'], 1)
        self.assertEqual(r['candidate_minus_baseline_means']['MRE-'], -2.5)
        self.assertIn('Descriptive only', r['secondary_objectives']['scope'])
        rows = [row for row in rows if row['criterion'] != 'absolute_error']
        self.assertEqual(compare_families(rows)['primary_comparison']['status'], 'insufficient_data')
        rows = fixture()
        for row in rows[:3]:
            row['MRE-'] = None
        r = compare_families(rows)
        self.assertIsNone(r['candidate_minus_baseline_means']['MRE-'])
        self.assertEqual(r['groups']['baseline']['objectives']['MRE-']['missing'], 3)

    def test_membership_independent_of_all_outcomes(self):
        rows = fixture()
        expected = select_families(rows, 'criterion', 'squared_error', 'absolute_error')
        for i, row in enumerate(rows):
            for name in ('MRE-', 'ACC+', 'PRED40+'):
                row[name] = None if i % 2 else 10000 - i
        self.assertEqual(select_families(rows, 'criterion', 'squared_error', 'absolute_error'), expected)
        r = compare_families(rows)
        self.assertEqual(r['groups']['baseline']['row_ids'], [1, 2, 3])
        self.assertEqual(r['groups']['candidate']['row_ids'], [4, 5, 6])

    def test_reproducibility_and_practical_separation(self):
        self.assertEqual(compare_families(fixture()), compare_families(fixture()))
        a = welch_comparison([10, 11, 12], [0, 1, 2])
        b = welch_comparison([10, 11, 12], [0, 1, 2], practical_threshold=1)
        c = welch_comparison([10, 11, 12], [0, 1, 2], practical_threshold=20)
        self.assertTrue(a['statistically_detected'])
        self.assertIsNone(a['practical_threshold'])
        self.assertIn('Undetermined', a['practical_assessment'])
        self.assertIn('supports improvement', b['practical_assessment'])
        self.assertIn('does not support', c['practical_assessment'])
        self.assertEqual(a['p_two_sided'], c['p_two_sided'])

    def test_actual_dataset_and_cli(self):
        _, rows = load_rows(DATASET)
        r = compare_families(rows)
        self.assertEqual(r['record_count'], 10000)
        for label, family in [('baseline', 'squared_error'), ('candidate', 'absolute_error')]:
            selected = [row for row in rows if row['criterion'] == family]
            self.assertEqual(r['groups'][label]['total_rows'], len(selected))
            for objective in ('MRE-', 'ACC+', 'PRED40+'):
                values = [row[objective] for row in selected if row[objective] is not None]
                self.assertAlmostEqual(r['groups'][label]['objectives'][objective]['mean'], sum(values) / len(values))
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
            output = Path(folder) / 'report.json'
            cmd = [sys.executable, str(ROOT / 'source/certify.py'), str(DATASET), '--output', str(output)]
            a = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(a.returncode, 0, a.stderr)
            first = output.read_bytes()
            b = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(b.returncode, 0, b.stderr)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(a.stdout, b.stdout)
            self.assertEqual(json.loads(first)['primary_comparison'], r['primary_comparison'])
            cmd[2] = str(Path(folder) / 'missing.csv')
            bad = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn('Error:', bad.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
