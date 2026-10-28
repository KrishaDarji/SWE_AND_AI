"""Independent synthetic score arithmetic and report contract checks."""
import copy
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
from spot import FIELDS, render_text, screen
from summarize import read_projects

DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


def fixture(values):
    return [{'Effort-': v} for v in values]


class SpotTests(unittest.TestCase):
    def test_independent_floor_and_direction(self):
        r = screen(fixture([0] * 9 + [1023]))
        top = r['ranked_findings'][0]
        self.assertEqual(top['csv_data_row'], 10)
        self.assertAlmostEqual(top['score'], 10)
        self.assertEqual(r['flagged_count'], 1)
        self.assertEqual(top['supporting_evidence'][0]['direction'], 'above')
        low = screen(fixture([1023] * 9 + [0]))['ranked_findings'][0]
        self.assertAlmostEqual(low['score'], 10)
        self.assertEqual(low['supporting_evidence'][0]['direction'], 'below')

    def test_independent_nonfloor_mad(self):
        # log1p outcomes are exactly 0,2,4,6,8: median=4, MAD=2.
        r = screen(fixture([math.expm1(v) for v in [0, 2, 4, 6, 8]]))
        self.assertAlmostEqual(r['reference']['Effort-']['log_median'], 4)
        self.assertAlmostEqual(r['reference']['Effort-']['scale'], 1.4826 * 2)
        self.assertAlmostEqual(r['ranked_findings'][0]['score'], 4 / (1.4826 * 2))
        self.assertEqual(r['flagged_count'], 0)

    def test_strict_boundary_and_max(self):
        rows = [{'Effort-': 0, 'Kloc+': 0, 'Defects-': 0} for _ in range(9)]
        rows.append({'Effort-': 1023, 'Kloc+': 3, 'Defects-': 7})
        score = screen(rows)['ranked_findings'][0]['score']
        self.assertEqual(screen(rows, score)['flagged_count'], 0)
        self.assertEqual(screen(rows, score - 0.001)['flagged_count'], 1)
        self.assertEqual(screen(rows, score + 0.001)['flagged_count'], 0)
        self.assertAlmostEqual(score, 10)

    def test_ties_constant_and_display(self):
        r = screen(fixture([5] * 5))
        self.assertEqual([x['csv_data_row'] for x in r['ranked_findings']], list(range(1, 6)))
        self.assertTrue(all(x['score'] == 0 and not x['flagged'] for x in r['ranked_findings']))
        before = copy.deepcopy(r)
        self.assertIn('Displaying 1 rows', render_text(r, 1))
        self.assertEqual(before, r)
        self.assertEqual(r['flagged_count'], 0)
        for bad in [-1, True, 1.5]:
            with self.assertRaises(ValueError):
                render_text(r, bad)

    def test_missing_and_partial_reference(self):
        rows = fixture([0, None, 0, 0, 1023])
        rows[0]['Kloc+'] = 99
        r = screen(rows)
        self.assertEqual(r['reference']['Effort-']['missing_count'], 1)
        self.assertFalse(r['reference']['Kloc+']['usable'])
        self.assertEqual(r['unscored_count'], 1)
        self.assertIsNone(r['ranked_findings'][-1]['score'])
        self.assertEqual(r['ranked_findings'][-1]['csv_data_row'], 2)
        self.assertFalse(r['ranked_findings'][-1]['flagged'])
        self.assertEqual(r['ranked_findings'][0]['csv_data_row'], 5)
        json.dumps(r, allow_nan=False)

    def test_invalid_and_small(self):
        for rows in [[], fixture([1]), fixture([1, 2]), [{}, {}, {}], fixture([None, 1, 2]), [None] * 3, 'bad']:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                screen(rows)
        for field in FIELDS:
            for bad in [-1, True, '1', float('nan'), float('inf')]:
                rows = fixture([1, 2, 3])
                rows[0][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    screen(rows)
        for threshold in [0, -1, True, None, '3', float('nan'), float('inf')]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                screen(fixture([1, 2, 3]), threshold)

    def test_metadata_category_exclusion_and_no_mutation(self):
        rows = fixture([0] * 9 + [1023])
        original = copy.deepcopy(rows)
        first = screen(rows)
        self.assertEqual(rows, original)
        for row in rows:
            row.update({'idX': -999, 'MonthsX': 1e100, 'unknownX': 999, 'prec': 'arbitrary unordered label'})
        changed = screen(rows)
        self.assertEqual(first['reference'], changed['reference'])
        self.assertEqual([(r['score'], r['flagged']) for r in first['ranked_findings']],
                         [(r['score'], r['flagged']) for r in changed['ranked_findings']])

    def test_reproducibility_and_cli(self):
        rows = read_projects(DATASET)
        r = screen(rows)
        self.assertEqual(r, screen(rows))
        self.assertEqual(len(r['ranked_findings']), len(rows))
        self.assertEqual(r['flagged_count'], sum(f['score'] is not None and f['score'] > 3.5 for f in r['ranked_findings']))
        for finding in r['ranked_findings']:
            self.assertEqual(finding['observed_outcomes'], {f: rows[finding['csv_data_row'] - 1][f] for f in FIELDS})
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            output = Path(tmp) / 'anomalies.json'
            command = [sys.executable, str(ROOT / 'source/spot.py'), str(DATASET), '--output', str(output)]
            p = subprocess.run(command + ['--limit', '1'], text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['ranked_findings'], r['ranked_findings'])
            self.assertEqual(len(saved['ranked_findings']), len(rows))
            p = subprocess.run(command + ['--limit', '-1'], text=True, capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('Display limit', p.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
