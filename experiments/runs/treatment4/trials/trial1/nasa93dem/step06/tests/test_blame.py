"""Hand-derived association checks and independent NASA93 raw CSV calculations."""
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from blame import diagnose
from summarize import PREDICTORS, read_projects
DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


def example():
    # Mean=3; total SS=20; prec between SS=16 and within SS=4: eta2=4/5.
    return [dict.fromkeys(PREDICTORS, 'constant') | {'prec': c, 'flex': c, 'team': d, 'Effort-': e}
            for c, d, e in [('a', 'x', 0), ('a', 'y', 2), ('b', 'y', 4), ('b', 'x', 6)]]


def factors(report):
    return {f['factor']: f for f in report['ranked_findings'] + report['unranked_factors']}


class DiagnosisTests(unittest.TestCase):
    def test_independent_arithmetic_direction_ties_and_constants(self):
        r = diagnose(example())
        f = factors(r)
        self.assertEqual([x['factor'] for x in r['ranked_findings']], ['flex', 'prec', 'team'])
        self.assertEqual([x['rank'] for x in r['ranked_findings']], [1, 1, 3])
        self.assertAlmostEqual(f['prec']['eta_squared'], 0.8)
        self.assertEqual(f['prec']['sums_of_squares'], {'between': 16, 'within': 4, 'total': 20})
        self.assertEqual(f['team']['eta_squared'], 0)
        self.assertEqual([c['observed_mean_direction'] for c in f['prec']['categories']], ['lower', 'higher'])
        self.assertEqual([c['mean_minus_eligible_mean'] for c in f['prec']['categories']], [-2, 2])
        self.assertEqual([c['effort_summary']['median'] for c in f['prec']['categories']], [1, 5])
        self.assertEqual([c['observed_mean_direction'] for c in f['team']['categories']], ['equal', 'equal'])
        self.assertEqual(f['pmat']['status'], 'constant_on_observed_rows')
        self.assertIsNone(f['pmat']['eta_squared'])
        self.assertIsNone(f['pmat']['rank'])
        self.assertIn('pmat', r['constant_factors'])
        self.assertTrue(all(c['small_category'] for c in f['prec']['categories']))

    def test_missing_values_pairwise_counts_and_normalization(self):
        rows = [{'prec': ' a ', 'Effort-': 1}, {'prec': 'a', 'Effort-': None},
                {'prec': 'b', 'Effort-': 3}, {'prec': ' ? ', 'Effort-': 100},
                {'prec': None, 'Effort-': None}, {'prec': 'NA', 'Effort-': 9}]
        r = diagnose(rows)
        p = factors(r)['prec']
        self.assertEqual(p['eligible_effort_summary']['count'], 2)
        self.assertEqual(p['eligible_effort_summary']['mean'], 2)
        self.assertEqual(p['eta_squared'], 1)
        self.assertEqual(r['missing_effort_count'], 2)
        self.assertEqual(p['excluded_missing_predictor_count'], 3)
        cats = {c['category']: c for c in p['categories']}
        self.assertEqual(cats['a']['project_count'], 2)
        self.assertEqual(cats['a']['missing_effort_count'], 1)
        self.assertEqual(cats[None]['observed_effort_count'], 2)
        self.assertEqual(cats[None]['effort_summary']['mean'], 54.5)
        self.assertFalse(cats[None]['included_in_score'])
        self.assertEqual(cats[None]['observed_mean_direction'], 'unavailable')
        self.assertEqual(factors(r)['flex']['status'], 'no_observed_categories')

    def test_zero_variance_singleton_and_empty_categories(self):
        r = diagnose([{'prec': 'a', 'Effort-': 2}, {'prec': 'b', 'Effort-': 2}])
        self.assertEqual(factors(r)['prec']['status'], 'zero_effort_variance')
        self.assertIsNone(factors(r)['prec']['eta_squared'])
        r = diagnose([{'prec': 'a', 'Effort-': 1}, {'prec': 'b', 'Effort-': None}])
        p = factors(r)['prec']
        self.assertFalse(p['constant_all_rows'])
        self.assertEqual(p['status'], 'constant_on_observed_rows')
        self.assertEqual(p['categories'][1]['effort_summary']['count'], 0)
        self.assertEqual(p['categories'][1]['observed_mean_direction'], 'unavailable')
        r = diagnose([{'Effort-': None}])
        self.assertEqual(r['observed_effort_summary']['count'], 0)
        self.assertEqual(r['ranked_findings'], [])
        json.dumps(r, allow_nan=False)

    def test_small_category_boundary(self):
        rows = [{'prec': 'a', 'Effort-': 1}] * 5 + [{'prec': 'b', 'Effort-': 2}] * 4
        cats = factors(diagnose(rows))['prec']['categories']
        self.assertFalse(cats[0]['small_category'])
        self.assertTrue(cats[1]['small_category'])

    def test_nominal_relabeling_preserves_strength_not_direction_of_names(self):
        original = example()
        changed = copy.deepcopy(original)
        for row in changed:
            row['prec'] = {'a': 'z_high_label', 'b': 'a_low_label'}[row['prec']]
        a, b = factors(diagnose(original))['prec'], factors(diagnose(changed))['prec']
        self.assertEqual(a['eta_squared'], b['eta_squared'])
        self.assertEqual(b['categories'][0]['observed_mean_direction'], 'higher')
        self.assertEqual(b['categories'][1]['observed_mean_direction'], 'lower')

    def test_reproducible_detached_no_mutation_and_exclusions(self):
        rows = example()
        saved = copy.deepcopy(rows)
        r = diagnose(rows)
        self.assertEqual(rows, saved)
        self.assertEqual(r, diagnose(rows))
        for row in rows:
            row.update({'Kloc+': 999, 'Defects-': 500, 'idX': 3, 'futureX': 'n', 'unknown': 7})
        self.assertEqual(r, diagnose(rows))
        self.assertEqual(set(factors(r)), set(PREDICTORS))
        r['ranked_findings'][0]['categories'].clear()
        self.assertNotEqual(r, diagnose(rows))

    def test_invalid_inputs(self):
        for rows in [[], [None], ['bad'], [{'prec': 2}], [{'prec': True}], [{'team': []}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                diagnose(rows)
        for target in [-1, True, '1', float('nan'), float('inf'), [], {}]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                diagnose([{'Effort-': target}])

    def test_independent_dataset_decimal_ranking_and_summaries(self):
        r = diagnose(read_projects(DATASET))
        with DATASET.open(newline='') as stream:
            raw = list(csv.DictReader(stream))
        scores = {}
        for field, item in factors(r).items():
            groups = defaultdict(list)
            for row in raw:
                groups[row[field].strip()].append(Decimal(row['Effort-']))
            values = [e for g in groups.values() for e in g]
            avg = sum(values) / len(values)
            ss = sum((e - avg) ** 2 for e in values)
            between = sum(len(g) * (sum(g) / len(g) - avg) ** 2 for g in groups.values())
            self.assertEqual(item['nonmissing_category_count_all_rows'], len(groups))
            if len(groups) > 1 and ss > 0:
                score = float(between / ss)
                scores[field] = score
                self.assertAlmostEqual(item['eta_squared'], score, places=12)
            else:
                self.assertIsNone(item['eta_squared'])
                self.assertIn(field, r['constant_factors'])
            for cat in item['categories']:
                g = sorted(groups[cat['category']])
                self.assertEqual(cat['observed_effort_count'], len(g))
                self.assertAlmostEqual(cat['effort_summary']['mean'], float(sum(g) / len(g)))
                n = len(g)
                med = (g[(n - 1) // 2] + g[n // 2]) / 2
                self.assertAlmostEqual(cat['effort_summary']['median'], float(med))
        expected = sorted(scores, key=lambda f: (-round(scores[f], 12), f))
        self.assertEqual([f['factor'] for f in r['ranked_findings']], expected)
        self.assertEqual(r['project_count'], 93)
        self.assertEqual(r, diagnose(read_projects(DATASET)))
        self.assertIn('not automatically excessive', ' '.join(r['limitations']))
        self.assertIn('eta squared', r['manager_explanation'])

    def test_cli_valid_and_invalid(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            out = Path(tmp) / 'blame.json'
            cmd = [sys.executable, str(ROOT / 'source/blame.py'), str(DATASET), '--output', str(out)]
            p = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads(out.read_text())
            self.assertEqual(r['dataset_path'], str(DATASET.resolve()))
            self.assertEqual(p.stdout, r['manager_explanation'])
            out.unlink()
            cmd[2] = str(Path(tmp) / 'missing.csv')
            p = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('Input error', p.stderr)
            self.assertFalse(out.exists())
            invalid = Path(tmp) / 'bad.csv'
            invalid.write_text('wrong,schema\n1,2\n')
            cmd[2] = str(invalid)
            p = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
