"""Independent arithmetic and edge cases for exploratory association diagnosis."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from blame import diagnose, partition
from summarize import CONFIGURATION, load_rows

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def fixture():
    return [dict(zip(CONFIGURATION, [x, c, 1, 0, x]), **{'MRE-': y})
            for x, c, y in [(1, 'z', 1), (1, 'z', 3), (2, 'a', 5), (2, 'a', 7)]]


def fields(report):
    return {f['field']: f for f in report['ranked_findings'] + report['unranked_fields']}


class BlameTests(unittest.TestCase):
    def test_independent_score_direction_ties_and_constants(self):
        r = diagnose(fixture())
        fs = fields(r)
        # grand mean=4; total SS=9+1+1+9=20; between SS=2*4+2*4=16.
        self.assertEqual(fs['N_estimators']['ss_total'], 20)
        self.assertEqual(fs['N_estimators']['ss_between'], 16)
        self.assertEqual(fs['N_estimators']['eta_squared'], .8)
        self.assertEqual([f['field'] for f in r['ranked_findings']],
                         ['N_estimators', 'criterion', 'Max_depth'])
        self.assertEqual([f['rank'] for f in r['ranked_findings']], [1, 1, 1])
        self.assertEqual(fs['N_estimators']['groups'][1]['observed_direction'], 'higher_MRE')
        self.assertEqual(fs['N_estimators']['groups'][0]['observed_direction'], 'lower_MRE')
        for name in ['Min_sample_leaves', 'Min_impurity_decrease']:
            self.assertEqual(fs[name]['status'], 'constant_predictor')
            self.assertIsNone(fs[name]['rank'])
            self.assertIsNone(fs[name]['eta_squared'])
        self.assertEqual(fs['criterion']['groups'][0]['definition'], {'value': 'a'})
        self.assertEqual(fs['criterion']['groups'][0]['mre']['mean'], 6)
        self.assertEqual(fs['criterion']['highest_mean_group_ids'], [0])
        self.assertTrue(all(g['sparse'] for g in fs['criterion']['groups']))

    def test_ranking_order_and_competition_ranks(self):
        rows = fixture()
        for row, x in zip(rows, [0, 1, 0, 1]):
            row['Min_sample_leaves'] = x
        fs = fields(diagnose(rows))
        # leaf group means 3 and 5: between SS=4, eta=.2; follows three ties at .8.
        self.assertEqual(fs['Min_sample_leaves']['eta_squared'], .2)
        self.assertEqual(fs['Min_sample_leaves']['rank'], 4)

    def test_categorical_relabeling_and_no_objective_predictors(self):
        rows = fixture()
        a = diagnose(rows)
        for row in rows:
            row['criterion'] = {'a': '999', 'z': '-400'}[row['criterion']]
            row['ACC+'] = 'unused'
            row['PRED40+'] = 99999
        b = diagnose(rows)
        self.assertEqual(fields(a)['criterion']['eta_squared'], fields(b)['criterion']['eta_squared'])
        self.assertEqual(a['predictor_fields'], list(CONFIGURATION))
        self.assertFalse(set(a['excluded_predictors']) & set(a['predictor_fields']))
        self.assertEqual(fields(b)['criterion']['grouping']['kind'], 'categories')

    def test_missing_values_counts_and_unlabeled_groups(self):
        rows = fixture()
        rows[0]['MRE-'] = None
        rows[1]['N_estimators'] = None
        rows.append({'N_estimators': 3, 'MRE-': None})
        f = fields(diagnose(rows))['N_estimators']
        self.assertEqual(f['status'], 'insufficient_labeled_groups')
        self.assertEqual(f['included_mre']['n'], 2)
        self.assertEqual(f['missing_predictor_mre']['n'], 1)
        self.assertEqual(f['groups'][0]['row_count'], 1)
        self.assertEqual(f['groups'][0]['mre']['missing'], 1)
        self.assertEqual(f['groups'][2]['observed_direction'], 'unavailable')
        all_missing = fields(diagnose([{'MRE-': None}]))
        self.assertTrue(all(f['status'] == 'all_missing_predictor' for f in all_missing.values()))
        rows = fixture()
        for row in rows:
            row['MRE-'] = None
        self.assertFalse(diagnose(rows)['ranked_findings'])

    def test_constant_outcome_and_equal_direction(self):
        rows = fixture()
        for row in rows:
            row['MRE-'] = 4
        f = fields(diagnose(rows))['N_estimators']
        self.assertEqual(f['status'], 'constant_outcome')
        self.assertIsNone(f['eta_squared'])
        self.assertTrue(all(g['observed_direction'] == 'equal' for g in f['groups']))
        for row, y in zip(rows, [1, 3, 1, 3]):
            row['MRE-'] = y
        f = fields(diagnose(rows))['N_estimators']
        self.assertEqual(f['eta_squared'], 0)
        self.assertEqual(f['highest_mean_group_ids'], [0, 1])
        self.assertEqual(f['lowest_mean_group_ids'], [0, 1])

    def test_hand_calculated_boundaries_and_boundary_membership(self):
        p, ids = partition(list(range(9)) + [None])
        self.assertEqual(p['cuts'], [2, 4, 6])
        self.assertEqual(ids, [0, 0, 0, 1, 1, 2, 2, 3, 3, None])
        self.assertEqual(p['groups'][0], {'lower': 0, 'upper': 2,
                                         'lower_inclusive': True, 'upper_inclusive': True})
        self.assertFalse(p['groups'][1]['lower_inclusive'])
        p, ids = partition(list(range(9)) + [8] * 30)
        self.assertEqual(p['cuts'], [])
        self.assertEqual(set(ids), {0})
        self.assertEqual(partition(list(range(8)))[0]['kind'], 'exact_values')
        rows = [{'N_estimators': i, 'MRE-': None if i % 2 else i} for i in range(9)]
        a = fields(diagnose(rows))['N_estimators']['grouping']
        for row in rows:
            row['MRE-'] = 999 - row['N_estimators']
        self.assertEqual(a, fields(diagnose(rows))['N_estimators']['grouping'])

    def test_joint_tables_show_perfect_cooccurrence(self):
        r = diagnose(fixture())
        pair = next(p for p in r['pairwise_group_summaries'] if p['fields'] == ['N_estimators', 'criterion'])
        self.assertEqual(len(r['pairwise_group_summaries']), 10)
        self.assertEqual([(c['group_ids'], c['row_count'], c['mre']['mean']) for c in pair['cells']],
                         [([0, 1], 2, 2), ([1, 0], 2, 6)])

    def test_reproducibility_normalization_and_no_mutation(self):
        rows = fixture()
        rows[0]['N_estimators'] = ' 1.0 '
        rows[0]['criterion'] = ' z '
        saved = copy.deepcopy(rows)
        a = diagnose(rows)
        self.assertEqual(a, diagnose(rows))
        self.assertEqual(rows, saved)
        self.assertEqual(a, diagnose(fixture()))
        self.assertEqual(a, diagnose(list(reversed(fixture()))))
        json.dumps(a, allow_nan=False)

    def test_sparse_threshold(self):
        rows = [{'criterion': 'a' if i < 5 else 'b', 'MRE-': i} for i in range(9)]
        g = fields(diagnose(rows))['criterion']['groups']
        self.assertFalse(g[0]['sparse'])
        self.assertTrue(g[1]['sparse'])

    def test_invalid_inputs(self):
        for rows in [None, {}, [], [None], [{}], [{'extra': 1, 'MRE-': 2}],
                     [{'MRE-': math.inf}], [{'MRE-': True}], [{'MRE-': 'bad'}],
                     [{'MRE-': 1, 'criterion': 2}], [{'MRE-': 1, 'Max_depth': math.nan}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                diagnose(rows)

    def test_actual_dataset_independent_ranking_calculations_and_cli(self):
        _, rows = load_rows(DATASET)
        r = diagnose(rows)
        self.assertEqual(r['record_count'], 10000)
        for f in r['ranked_findings']:
            name = f['field']
            cuts = f['grouping']['cuts']
            buckets = {}
            for row in rows:
                x, y = row[name], row['MRE-']
                if x is None or y is None:
                    continue
                key = sum(x > cut for cut in cuts) if cuts else x
                buckets.setdefault(key, []).append(y)
            ys = [y for group in buckets.values() for y in group]
            mu = sum(ys) / len(ys)
            total = sum((y - mu) ** 2 for y in ys)
            between = sum(len(g) * (sum(g) / len(g) - mu) ** 2 for g in buckets.values())
            self.assertAlmostEqual(f['eta_squared'], between / total, places=12)
            self.assertEqual(f['included_mre']['n'], len(ys))
        scores = [f['eta_squared'] for f in r['ranked_findings']]
        self.assertEqual(scores, sorted(scores, reverse=True))
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            output = Path(directory) / 'blame.json'
            cmd = [sys.executable, str(ROOT / 'source/blame.py'), str(DATASET), '--output', str(output)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            content = output.read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(content, output.read_bytes())
            self.assertEqual(first.stdout, second.stdout)
            cmd[2] = str(Path(directory) / 'absent.csv')
            bad = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn('Error:', bad.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
