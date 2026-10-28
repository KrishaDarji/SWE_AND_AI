"""Controlled screening examples and audit checks; no detection-accuracy claims."""

import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
from remember import Measurements, load_measurements
import spot


def data(rows):
    return Measurements(('setting',), tuple(tuple(row) for row in rows))


class SpotTests(unittest.TestCase):
    def test_hand_calculated_scores_and_threshold(self):
        result = spot.screen(data([(0, 0), (1, 1), (2, 9)]), neighbors=1)
        rows = {f['data_row']: f for f in result['ranked_findings']}
        # Middle point has equal-distance neighbors; both participate: median(0,9)=4.5.
        for i, expected in enumerate([1, 3.5, 8], 1):
            self.assertAlmostEqual(rows[i]['score'], expected)
        # median=3.5, MAD=2.5; scale=1.4826*2.5, exceeds .01*9.
        self.assertAlmostEqual(result['threshold'], 3.5 + 3 * 1.4826 * 2.5)
        self.assertEqual(rows[2]['evidence']['neighbor_data_rows'], [1, 3])
        self.assertEqual(result['flagged_count'], 0)

    def test_threshold_equality_is_not_flagged(self):
        # Scores [1,0,1], median 1, MAD 0, threshold >1 due to range floor.
        result = spot.screen(data([(0, 0), (1, 1), (2, 2)]), neighbors=1)
        self.assertEqual(result['flagged_count'], 0)
        constant = spot.screen(data([(0, 4), (1, 4), (2, 4)]))
        self.assertEqual(constant['threshold'], 0)
        self.assertTrue(all(f['score'] == 0 and not f['flagged'] for f in constant['ranked_findings']))

    def test_spike_high_and_low_and_floor(self):
        for spike in (100, -100):
            result = spot.screen(data([(i, spike if i == 10 else 0) for i in range(21)]), neighbors=4)
            flagged = [f for f in result['ranked_findings'] if f['flagged']]
            self.assertEqual([f['data_row'] for f in flagged], [11])
            self.assertAlmostEqual(result['threshold'], 3)
            self.assertEqual(result['threshold_components']['mad_score'], 0)
            self.assertAlmostEqual(flagged[0]['score'], 100)
            self.assertTrue(flagged[0]['suggested_checks'])

    def test_duplicate_configurations_excluded_and_group_weighted(self):
        result = spot.screen(data([(0, 100), (0, 100), (1, 0), (2, 2), (3, 4)]), neighbors=2)
        row = next(f for f in result['ranked_findings'] if f['data_row'] == 1)
        self.assertEqual(row['evidence']['excluded_configuration_data_rows'], [1, 2])
        self.assertEqual(row['evidence']['neighbor_data_rows'], [3, 4])
        self.assertAlmostEqual(row['evidence']['local_median_Performance-'], 1)
        for f in result['ranked_findings']:
            self.assertTrue(set(f['evidence']['neighbor_data_rows']).isdisjoint(
                f['evidence']['excluded_configuration_data_rows']))

    def test_ties_competition_rank_and_stable_row_order(self):
        result = spot.screen(data([(0, 7), (1, 7), (2, 7), (3, 7)]))
        self.assertEqual([f['rank'] for f in result['ranked_findings']], [1]*4)
        self.assertEqual([f['data_row'] for f in result['ranked_findings']], [1, 2, 3, 4])

    def test_unequal_replicate_counts_do_not_weight_neighbors(self):
        sample = data([(0, 0), (1, 10), (1, 10), (1, 10), (2, 20)])
        report = spot.screen(sample, neighbors=2)
        row = next(f for f in report['ranked_findings'] if f['data_row'] == 1)
        self.assertAlmostEqual(row['evidence']['local_median_Performance-'], 15)
        self.assertFalse(row['evidence']['within_neighbor_configuration_median_range'])
        self.assertAlmostEqual(row['evidence']['neighbor_configuration_median_MAD'], 5)

    def test_evaluated_outcome_does_not_change_own_local_prediction(self):
        a = spot.screen(data([(0, 1), (0, 2), (1, 3), (2, 5), (3, 8)]), neighbors=2)
        b = spot.screen(data([(0, 100), (0, 200), (1, 3), (2, 5), (3, 8)]), neighbors=2)
        for report in (a, b):
            row = next(f for f in report['ranked_findings'] if f['data_row'] == 1)
            self.assertAlmostEqual(row['evidence']['local_median_Performance-'], 4)

    def test_custom_multiplier_changes_threshold_not_scores(self):
        sample = data([(i, i*i) for i in range(12)])
        a = spot.screen(sample, neighbors=3, multiplier=1)
        b = spot.screen(sample, neighbors=3, multiplier=4)
        self.assertLess(a['threshold'], b['threshold'])
        self.assertEqual([f['score'] for f in a['ranked_findings']],
                         [f['score'] for f in b['ranked_findings']])
        for report in (a,b):
            self.assertTrue(all(f['flagged'] == (f['score'] > report['threshold'])
                                for f in report['ranked_findings']))

    def test_reproducibility_and_display_independence(self):
        sample = data([(i, i*i) for i in range(20)])
        a, b = spot.screen(sample), spot.screen(sample)
        self.assertEqual(a, b)
        before = json.dumps(a, sort_keys=True)
        self.assertEqual(len(spot.render_text(a, 1).splitlines()), 3)
        self.assertEqual(len(spot.render_text(a, 5).splitlines()), 7)
        self.assertEqual(json.dumps(a, sort_keys=True), before)
        self.assertIsNone(a['parameters']['seed'])

    def test_setting_range_scaling_and_constants(self):
        a = spot.screen(data([(i, i*i) for i in range(8)]), neighbors=2)
        b = spot.screen(Measurements(('scaled', 'constant'),
                        tuple((100*i + 9, 42, i*i) for i in range(8))), neighbors=2)
        self.assertEqual(b['input_fields']['constant_fields_excluded_from_distance'], ['constant'])
        self.assertEqual([f['score'] for f in a['ranked_findings']],
                         [f['score'] for f in b['ranked_findings']])

    def test_invalid_and_small_inputs(self):
        for sample in (None, data([]), data([(0, 1)]), data([(0, 1), (0, 2), (1, 3)]),
                       data([(0, 1), (1, math.nan), (2, 3)]),
                       data([(0, 1), (1, True), (2, 3)]),
                       Measurements(('x','x'), ((0,0,1),)),
                       Measurements(('x',), ((0,1,2),))):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                spot.screen(sample)
        valid = data([(0, 0), (1, 1), (2, 2)])
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                spot.screen(valid, neighbors=k)
        for m in (0, -1, True, float('inf'), float('nan'), '3'):
            with self.assertRaises(ValueError):
                spot.screen(valid, multiplier=m)
        for top in (0, True, -1):
            with self.assertRaises(ValueError):
                spot.render_text(spot.screen(valid), top)
        self.assertEqual(spot.screen(valid)['parameters']['effective_neighbors_before_ties'], 2)

    def test_cli_and_input_protection(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            root = Path(directory)
            dataset = root / 'input.csv'
            dataset.write_text('x,Performance-\n0,1\n1,2\n2,3\n')
            before = dataset.read_bytes()
            command = [sys.executable, '-B', str(STEP/'source/spot.py'), str(dataset),
                       '--output-dir', str(root), '--top', '1']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root/'anomalies.json').read_text())
            self.assertEqual(len(result['ranked_findings']), 3)
            self.assertEqual(result['display_limit'], 1)
            self.assertEqual(dataset.read_bytes(), before)
            protected = root/'anomalies.json'
            protected.write_bytes(before)
            run = subprocess.run([sys.executable, '-B', str(STEP/'source/spot.py'), str(protected),
                                  '--output-dir', str(root)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(protected.read_bytes(), before)
            bad = subprocess.run(command + ['--top','0'], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)

    def test_dataset_audit_and_independent_threshold(self):
        path = os.environ.get('X264_DATASET')
        if not path:
            self.skipTest('Set X264_DATASET for supplied-dataset audit')
        source = load_measurements(path)
        report = spot.screen(source)
        self.assertEqual(report['row_count'], len(source.rows))
        scores = [f['score'] for f in report['ranked_findings']]
        center = statistics.median(scores)
        mad = statistics.median(abs(v-center) for v in scores)
        targets = [r[-1] for r in source.rows]
        threshold = center + 3*max(1.4826*mad, .01*(max(targets)-min(targets)))
        self.assertAlmostEqual(report['threshold'], threshold)
        self.assertEqual(len({f['data_row'] for f in report['ranked_findings']}), len(source.rows))
        for f in report['ranked_findings']:
            original = source.rows[f['data_row']-1]
            self.assertEqual(f[ 'Performance-'], original[-1])
            self.assertEqual(tuple(f['settings'].values()), original[:-1])
            self.assertEqual(f['flagged'], f['score'] > report['threshold'])
            own = set(f['evidence']['excluded_configuration_data_rows'])
            self.assertTrue(own.isdisjoint(f['evidence']['neighbor_data_rows']))
        self.assertEqual(report['flagged_count'], sum(f['flagged'] for f in report['ranked_findings']))


if __name__ == '__main__':
    unittest.main()
