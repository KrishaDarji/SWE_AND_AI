"""Independent exact examples, validation, fixed-seed checks and dataset integration."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
import certify
from remember import Measurements, load_measurements


class CertificationTests(unittest.TestCase):
    def test_exact_improvement_direction(self):
        # Among C(8,4)=70 allocations, only the two fully separated allocations
        # attain absolute mean difference 4. Thus the two-sided p-value is 2/70.
        result = certify.compare_samples([5, 6, 7, 8], [1, 2, 3, 4])
        self.assertEqual((result['baseline_n'], result['candidate_n']), (4, 4))
        self.assertEqual(result['difference_candidate_minus_baseline'], -4)
        self.assertEqual(result['observed_improvement'], 4)
        self.assertEqual(result['draws'], 70)
        self.assertEqual(result['extreme_assignments'], 2)
        self.assertAlmostEqual(result['p_value'], 2 / 70)
        self.assertTrue(result['credible_improvement_conditional_on_assumptions'])
        reverse = certify.compare_samples([1, 2, 3, 4], [5, 6, 7, 8])
        self.assertEqual(reverse['p_value'], result['p_value'])
        self.assertFalse(reverse['credible_improvement_conditional_on_assumptions'])
        self.assertIn('unfavorable', reverse['conclusion'])
        maximize = certify.compare_samples([1, 2, 3, 4], [5, 6, 7, 8], direction='maximize')
        self.assertTrue(maximize['credible_improvement_conditional_on_assumptions'])

    def test_identical_samples_are_not_equivalence(self):
        for values in ([1, 2, 3, 4], [7, 7, 7, 7]):
            result = certify.compare_samples(values, values)
            self.assertEqual(result['p_value'], 1)
            self.assertEqual(result['observed_improvement'], 0)
            self.assertFalse(result['credible_improvement_conditional_on_assumptions'])
            self.assertIn('not proof of equivalence', result['conclusion'])

    def test_small_sample_exact_resolution(self):
        result = certify.compare_samples([4, 4], [1, 1])
        self.assertEqual(result['draws'], 6)
        self.assertAlmostEqual(result['p_value'], 2 / 6)
        self.assertFalse(result['statistically_detected_difference'])

    def test_practical_threshold_is_separate(self):
        result = certify.compare_samples([5, 6, 7, 8], [1, 2, 3, 4], min_effect=5)
        self.assertTrue(result['statistically_detected_difference'])
        self.assertFalse(result['observed_practical_improvement'])
        self.assertFalse(result['credible_improvement_conditional_on_assumptions'])

    def test_unequal_sizes_ties_and_manual_count(self):
        # Pool 0,0,0,3,3. The baseline draws 2. Three choices of two zeros
        # have |difference|=2, six mixed choices have .5, one pair of threes
        # has 3. Observed baseline (0,0), candidate (0,3,3): p=(3+1)/10.
        result = certify.compare_samples([0, 0], [0, 3, 3])
        self.assertEqual(result['draws'], 10)
        self.assertEqual(result['extreme_assignments'], 4)
        self.assertAlmostEqual(result['p_value'], .4)

    def test_invalid_samples_and_options(self):
        for invalid in ([], [1], [1, float('nan')], [1, float('inf')],
                        [1, '2'], [True, 1], [None, 1], [1j, 2]):
            for a, b in ((invalid, [1, 2]), ([1, 2], invalid)):
                with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                    certify.compare_samples(a, b)
        for options in ({'direction': 'lower'}, {'alpha': 0}, {'alpha': 1},
                        {'alpha': float('nan')}, {'alpha': True},
                        {'min_effect': -1}, {'min_effect': float('inf')},
                        {'seed': -1}, {'seed': True}, {'resamples': 0},
                        {'resamples': 2.5}, {'resamples': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                certify.compare_samples([1, 2], [3, 4], **options)

    def test_monte_carlo_reproducibility_and_plus_one(self):
        a, b = list(range(10)), list(range(10, 20))
        first = certify.compare_samples(a, b, resamples=499)
        self.assertEqual(first, certify.compare_samples(a, b, resamples=499))
        self.assertIn('Monte Carlo', first['method'])
        self.assertEqual(first['seed'], 42031)
        self.assertEqual(first['draws'], 499)
        self.assertEqual(first['p_value'], (first['extreme_assignments'] + 1) / 500)
        self.assertGreater(first['p_value'], 0)
        self.assertEqual(first['observed_improvement'], -10)

    def test_group_rule_independent_of_objective(self):
        data = Measurements(('no_8x8dct', 'other'),
                            ((0., 0., 5.), (1., 1., 2.), (0., 1., 7.), (1., 0., 4.)))
        result = certify.dataset_demo(data)
        self.assertEqual(result['comparison']['baseline_mean'], 6)
        self.assertEqual(result['comparison']['candidate_mean'], 3)
        self.assertEqual(result['comparison']['baseline_n'], 2)
        changed = Measurements(data.settings, tuple((*r[:-1], -r[-1]) for r in data.rows))
        self.assertEqual(result['group_selection_rule'],
                         certify.dataset_demo(changed)['group_selection_rule'])
        for invalid in (Measurements(('x',), ((0., 2.), (1., 3.))),
                        Measurements(('no_8x8dct',), ((0., 2.), (2., 3.)))):
            with self.assertRaises(ValueError):
                certify.dataset_demo(invalid)

    def test_dataset_cli_and_independent_means(self):
        dataset = Path(os.environ['X264_DATASET'])
        before = dataset.read_bytes()
        with dataset.open(newline='') as f:
            rows = list(csv.DictReader(f))
        groups = [[float(r['Performance-']) for r in rows
                   if float(r['no_8x8dct']) == value] for value in (0, 1)]
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            process = subprocess.run([sys.executable, '-B', str(STEP / 'source/certify.py'),
                                      str(dataset), '--output-dir', tmp, '--resamples', '99'],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads((Path(tmp) / 'certification.json').read_text())
            self.assertEqual(json.loads(process.stdout), result)
            comparison = result['comparison']
            self.assertEqual([comparison['baseline_n'], comparison['candidate_n']],
                             [len(g) for g in groups])
            self.assertAlmostEqual(comparison['baseline_mean'], sum(groups[0]) / len(groups[0]))
            self.assertAlmostEqual(comparison['candidate_mean'], sum(groups[1]) / len(groups[1]))
            self.assertEqual(result['quality_measures'], [])
            self.assertIsNone(result['units'])
        self.assertEqual(dataset.read_bytes(), before)

    def test_cli_prevents_overwrite_and_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            dataset = Path(tmp) / 'certification.json'
            dataset.write_text('no_8x8dct,Performance-\n0,3\n0,4\n1,1\n1,2\n')
            before = dataset.read_bytes()
            with self.assertRaises(SystemExit) as result:
                certify.main([str(dataset), '--output-dir', tmp])
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(before, dataset.read_bytes())
            dataset.write_text('no_8x8dct,Performance-\n0,nan\n')
            with self.assertRaises(SystemExit) as result:
                certify.main([str(dataset), '--output-dir', tmp])
            self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
