"""Independent arithmetic, boundary cases, observed-data and CLI checks."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from certify import certify, compare_effort
DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


class ComparisonTests(unittest.TestCase):
    def test_hand_calculated_welch(self):
        r = compare_effort([1, 3], [0, 2])
        s = r['statistical']
        self.assertEqual(r['difference_candidate_minus_baseline'], -1)
        self.assertAlmostEqual(s['standard_error'], math.sqrt(2))
        self.assertAlmostEqual(s['degrees_of_freedom'], 2)
        self.assertAlmostEqual(s['t_statistic'], -1 / math.sqrt(2))
        # For df=2 the two-sided p is 1-|t|/sqrt(t^2+2).
        self.assertAlmostEqual(s['p_value_two_sided'], 1 - 1 / math.sqrt(5))
        # Independent published/tabulated 97.5% t quantile at df=2.
        self.assertAlmostEqual(s['mean_difference_ci95'][1], -1 + 4.3026527297 * math.sqrt(2), places=7)
        self.assertTrue(r['small_sample_warning'])

    def test_improvement_and_reverse(self):
        a, b = [100, 101, 102, 103, 104], [10, 11, 12, 13, 14]
        better, worse = compare_effort(a, b), compare_effort(b, a)
        self.assertEqual(better['observed_direction'], 'lower')
        self.assertEqual(better['statistical']['evidence'], 'candidate_lower')
        self.assertTrue(better['practical']['supported_by_ci95'])
        self.assertEqual(worse['statistical']['evidence'], 'candidate_higher')
        self.assertFalse(worse['practical']['supported_by_ci95'])
        self.assertAlmostEqual(better['statistical']['p_value_two_sided'], worse['statistical']['p_value_two_sided'])

    def test_identical_samples(self):
        r = compare_effort([1, 2, 3], [1, 2, 3])
        self.assertEqual(r['statistical']['p_value_two_sided'], 1)
        self.assertEqual(r['observed_direction'], 'equal')
        self.assertIn('not proof of equivalence', r['manager_conclusion'])
        self.assertFalse(r['practical']['observed_threshold_met'])

    def test_degenerate_and_zero_baseline(self):
        for a, b in [([2, 2], [2, 2]), ([2, 2], [1, 1]), ([0, 0], [0, 0])]:
            r = compare_effort(a, b)
            self.assertIsNone(r['statistical']['p_value_two_sided'])
            self.assertFalse(r['practical']['supported_by_ci95'])
            json.dumps(r, allow_nan=False)
        self.assertIsNone(compare_effort([0, 0], [1, 2])['relative_mean_reduction'])

    def test_small_invalid_samples(self):
        for a in [[], [1], [None, 1], [-1, 2], [float('nan'), 2], [float('inf'), 2], ['2', 3], [True, 2]]:
            for left, right in [(a, [1, 2]), ([1, 2], a)]:
                with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                    compare_effort(left, right)
        for f in [0, 1, -1, float('nan'), 'bad', True]:
            with self.subTest(f=f), self.assertRaises(ValueError):
                compare_effort([1, 2], [3, 4], f)

    def test_missing_and_reproducibility(self):
        r = compare_effort([None, 1, 2, 3], [0, None, 1, 2])
        clean = compare_effort([1, 2, 3], [0, 1, 2])
        self.assertEqual(r['baseline']['count'], 3)
        self.assertEqual(r['baseline']['excluded_missing_outcomes'], 1)
        self.assertEqual(r['candidate']['excluded_missing_outcomes'], 1)
        self.assertEqual(r['statistical'], clean['statistical'])
        self.assertEqual(r['bootstrap_sensitivity'], clean['bootstrap_sensitivity'])
        self.assertEqual(r, compare_effort([None, 1, 2, 3], [0, None, 1, 2]))
        self.assertEqual(r['bootstrap_sensitivity']['seed'], 42)

    def test_statistical_vs_practical(self):
        r = compare_effort([100, 100.01, 100.02], [99, 99.01, 99.02])
        self.assertEqual(r['statistical']['evidence'], 'candidate_lower')
        self.assertFalse(r['practical']['observed_threshold_met'])
        self.assertFalse(r['practical']['supported_by_ci95'])

    def test_independent_dataset_and_cli(self):
        r = certify(DATASET)
        with DATASET.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for label, level in [('baseline', 'n'), ('candidate', 'h')]:
            values = [float(row['Effort-']) for row in rows if row['tool'].strip() == level]
            self.assertEqual(r[label]['count'], len(values))
            self.assertAlmostEqual(r[label]['mean'], sum(values) / len(values))
        self.assertEqual(r['group_definitions']['baseline'], 'tool=n')
        self.assertEqual(r['group_definitions']['candidate'], 'tool=h')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            out = Path(tmp) / 'certification.json'
            p = subprocess.run([sys.executable, str(ROOT / 'source/certify.py'), str(DATASET), '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(out.read_text()), r)


if __name__ == '__main__':
    unittest.main(verbosity=2)
