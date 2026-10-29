"""Hand-checkable synthetic contracts for descriptive anomaly screening."""
import copy
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from spot import crosses_threshold, reference, score_value, screen, screen_rows
from summarize import CONFIGURATION, OBJECTIVES

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


class SpotTests(unittest.TestCase):
    def test_hand_calculated_scores_and_both_tails(self):
        # Sorted values: median=0, deviations median=1; extremes score 10/1.4826.
        values = [-10, -1, -1, 0, 1, 1, 10]
        rows = [{'MRE-': x, 'ACC+': x, 'PRED40+': x} for x in values]
        report = screen_rows(rows)
        self.assertEqual(report['flagged_count'], 2)
        self.assertEqual([r['row_id'] for r in report['ranked_findings'][:2]], [1, 7])
        self.assertAlmostEqual(report['ranked_findings'][0]['anomaly_score'], 10 / 1.4826)
        e = report['ranked_findings'][0]['supporting_evidence']
        self.assertEqual(e['MRE-']['relation_to_median_in_objective_direction'], 'better')
        self.assertEqual(e['ACC+']['relation_to_median_in_objective_direction'], 'worse')
        self.assertEqual(report['references']['MRE-']['median'], 0)
        self.assertEqual(report['references']['MRE-']['mad'], 1)
        self.assertEqual(report['references']['MRE-']['iqr'], 2)

    def test_strict_threshold_boundary_and_not_quota(self):
        self.assertFalse(crosses_threshold(None, 3.5))
        self.assertFalse(crosses_threshold(3.5, 3.5))
        self.assertTrue(crosses_threshold(math.nextafter(3.5, math.inf), 3.5))
        rows = [{'MRE-': x} for x in [-10, -1, -1, 0, 1, 1, 10]]
        exact = 10 / 1.4826
        self.assertEqual(screen_rows(rows, exact)['flagged_count'], 0)
        self.assertEqual(screen_rows(rows, math.nextafter(exact, 0))['flagged_count'], 2)

    def test_constants_and_small_inputs(self):
        for n in (1, 4, 5, 9):
            r = screen_rows([{'MRE-': 8}] * n)
            self.assertEqual(r['flagged_count'], 0)
            self.assertEqual(r['unscored_count'], n if n < 5 else 0)
            self.assertEqual(r['ranked_findings'][0]['anomaly_score'], None if n < 5 else 0)
        self.assertEqual(reference([8] * 5)['scale_method'], 'constant')
        self.assertIsNone(reference([])['median'])

    def test_degenerate_scale_fallbacks(self):
        # median=0, MAD=0; q75=.75, q25=0 => IQR=.75.
        r = reference([0, 0, 0, 0, 1, 2])
        self.assertEqual(r['scale_method'], 'IQR / 1.349 fallback')
        self.assertAlmostEqual(r['scale'], .75 / 1.349)
        r = reference([0] * 8 + [10])
        self.assertEqual(r['scale_method'], 'observed range fallback')
        self.assertEqual(score_value(10, r), 1)

    def test_missing_objectives_and_max_combination(self):
        rows = [{'MRE-': x, 'ACC+': None, 'Max_depth': ''}
                for x in [-10, -1, -1, 0, 1, 1, 10]] + [{}]
        r = screen_rows(rows)
        self.assertEqual(r['missing_data_handling']['counts']['ACC+'], 8)
        self.assertEqual(r['missing_data_handling']['counts']['MRE-'], 1)
        self.assertEqual(r['ranked_findings'][-1]['row_id'], 8)
        self.assertIsNone(r['ranked_findings'][-1]['anomaly_score'])
        self.assertEqual(r['flagged_count'], 2)
        for f in r['ranked_findings']:
            scores = [e['score'] for e in f['supporting_evidence'].values() if e['score'] is not None]
            self.assertEqual(f['anomaly_score'], max(scores) if scores else None)
            self.assertEqual(f['crosses_threshold'], any(e['crosses_threshold'] for e in f['supporting_evidence'].values()))

    def test_configuration_is_context_not_score_input(self):
        rows = [{'MRE-': x, 'N_estimators': ' 10 ', 'criterion': ' a '}
                for x in [-10, -1, -1, 0, 1, 1, 10]]
        a = screen_rows(rows)
        for i, row in enumerate(rows):
            row['N_estimators'] = i
            row['criterion'] = str(i)
        b = screen_rows(rows)
        self.assertEqual([f['anomaly_score'] for f in a['ranked_findings']],
                         [f['anomaly_score'] for f in b['ranked_findings']])
        context = a['ranked_findings'][0]['same_configuration_context']
        self.assertEqual(context['record_count'], 7)
        self.assertEqual(context['observed_objective_ranges']['MRE-'],
                         {'observed_count': 7, 'min': -10, 'max': 10})

    def test_ties_reproducibility_detachment(self):
        rows = [{'MRE-': x} for x in [-10, -1, -1, 0, 1, 1, 10]]
        saved = copy.deepcopy(rows)
        a, b = screen_rows(rows), screen_rows(rows)
        self.assertEqual(a, b)
        self.assertEqual(rows, saved)
        self.assertEqual([r['row_id'] for r in a['ranked_findings']], [1, 7, 2, 3, 5, 6, 4])
        a['ranked_findings'][0]['configuration']['Max_depth'] = 100
        self.assertEqual(rows, saved)
        json.dumps(b, allow_nan=False)

    def test_invalid_inputs(self):
        for bad in ([], None, {}, [1], [{'unknown': 1}], [{'MRE-': True}],
                    [{'ACC+': math.nan}], [{'MRE-': 'bad'}], [{'criterion': 5}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                screen_rows(bad)
        for bad in (True, 0, -1, math.inf, math.nan, '3'):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                screen_rows([{}], bad)
        with self.assertRaises(ValueError):
            screen_rows([{'MRE-': x} for x in [-1e308, -1e308, 0, 1e308, 1e308]])

    def test_real_csv_independent_score_calculations(self):
        with DATASET.open(newline='') as handle:
            raw = list(csv.DictReader(handle))
        r = screen(DATASET)
        refs = {}
        for name in OBJECTIVES:
            xs = [float(row[name]) for row in raw if row[name].strip()]
            median = statistics.median(xs)
            scale = 1.4826 * statistics.median(abs(x - median) for x in xs)
            if scale == 0:
                quartiles = statistics.quantiles(xs, n=4, method='inclusive')
                iqr = quartiles[2] - quartiles[0]
                scale = iqr / 1.349 if iqr > 0 else max(xs) - min(xs)
            refs[name] = median, scale
        expected = []
        for i, row in enumerate(raw, 1):
            score = max(abs(float(row[n]) - refs[n][0]) / refs[n][1] if refs[n][1] else 0 for n in OBJECTIVES)
            expected.append((score, i))
        expected.sort(key=lambda p: (-p[0], p[1]))
        self.assertEqual(len(r['ranked_findings']), len(raw))
        for f, (score, row_id) in zip(r['ranked_findings'], expected):
            self.assertEqual(f['row_id'], row_id)
            self.assertAlmostEqual(f['anomaly_score'], score)
        self.assertEqual(r['flagged_count'], sum(s > 3.5 for s, _ in expected))

    def test_cli_display_is_separate_and_csv_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            path, output = Path(directory) / 'data.csv', Path(directory) / 'report.json'
            with path.open('w', newline='') as handle:
                w = csv.writer(handle)
                w.writerow([*OBJECTIVES, *CONFIGURATION])
                for x in [-10, -1, -1, 0, 1, 1, 10]:
                    w.writerow([x, '', '', '', '', '', '', ''])
            cmd = [sys.executable, str(ROOT / 'source/spot.py'), str(path), '--output', str(output)]
            for limit in (0, 1, 20):
                result = subprocess.run(cmd + ['--display-limit', str(limit)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                console, report = json.loads(result.stdout), json.loads(output.read_text())
                self.assertEqual(len(console['displayed_findings']), min(limit, 7))
                self.assertEqual(report['flagged_count'], 2)
                self.assertEqual(len(report['ranked_findings']), 7)
            for args in (['--threshold', 'nan'], ['--display-limit', '-1']):
                result = subprocess.run(cmd + args, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
            for text in ('', 'MRE-,MRE-\n1,2\n', ','.join([*OBJECTIVES, *CONFIGURATION]) + '\n',
                         ','.join([*OBJECTIVES, *CONFIGURATION]) + '\nnan,1,1,1,a,1,1,1\n',
                         ','.join([*OBJECTIVES, *CONFIGURATION]) + '\n1\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    screen(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
