"""Constrained selection contracts and independent exhaustive CSV verification."""
import copy
import csv
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from choose import choose, render_text
from summarize import COLUMNS, PREDICTORS, read_projects

DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


def row(effort, value, label='n'):
    return dict.fromkeys(PREDICTORS, label) | {'Effort-': effort, 'Kloc+': value, 'Defects-': None, 'idX': 1}


class SelectionTests(unittest.TestCase):
    def test_maximum_value_and_budget_feasibility(self):
        rows = [row(101, 1000), row(95, 20), row(99, 80), row(10, 5)]
        report = choose(rows, 100)
        chosen = report['selected_project']
        self.assertEqual(chosen['csv_data_row'], 3)
        self.assertEqual(chosen['observed_outcomes']['Kloc+'], 80)
        self.assertEqual(chosen['feasibility']['remaining_effort_budget'], 1)
        self.assertTrue(chosen['feasibility']['satisfied'])
        self.assertEqual(report['counts']['feasible_candidates'], 3)
        self.assertEqual(report['counts']['over_budget_candidates'], 1)

    def test_exact_boundary_no_tolerance_and_zero(self):
        rows = [row(100, 5), row(math.nextafter(100, math.inf), 99)]
        self.assertEqual(choose(rows, 100)['selected_project']['csv_data_row'], 1)
        self.assertIsNone(choose(rows, math.nextafter(100, -math.inf))['selected_project'])
        self.assertEqual(choose([row(0, 0), row(1, 100)], 0)['selected_project']['csv_data_row'], 1)

    def test_ties_lower_effort_then_earlier_data_row_duplicate_id(self):
        rows = [row(60, 50), row(20, 50, 'h'), row(20, 50, 'l')]
        rows[1]['idX'] = None
        report = choose(rows, 100)
        self.assertEqual(report['selected_project']['csv_data_row'], 2)
        self.assertEqual(report['selected_project']['characteristics']['prec'], 'h')
        self.assertEqual(choose(rows, 100), report)
        rows[1]['idX'] = rows[2]['idX'] = 999
        self.assertEqual(choose(rows, 100)['selected_project']['csv_data_row'], 2)

    def test_missing_outcomes_union_counts_and_original_row_identity(self):
        rows = [row(None, 10), row(3, None), row(None, None), row(5, 8)]
        result = choose(rows, 100)
        self.assertEqual(result['counts'], {
            'total_rows': 4, 'eligible_rows': 1, 'excluded_missing_required_outcomes': 3,
            'missing_by_required_outcome': {'Effort-': 2, 'Kloc+': 2},
            'feasible_candidates': 1, 'over_budget_candidates': 0})
        self.assertEqual(result['selected_project']['csv_data_row'], 4)
        self.assertIsNone(result['selected_project']['observed_outcomes']['Defects-'])
        self.assertEqual([r['csv_data_row'] for r in result['excluded_rows']], [1, 2, 3])

    def test_no_feasible_candidates_and_empty(self):
        for rows in [[], [row(101, 20)], [row(None, 20)], [row(10, None)]]:
            with self.subTest(rows=rows):
                report = choose(rows, 100)
                self.assertEqual(report['status'], 'no_feasible_candidate')
                self.assertIsNone(report['selected_project'])
                self.assertEqual(report['counts']['feasible_candidates'], 0)
                self.assertIn('No feasible candidate', report['readable_explanation'])
                self.assertEqual(report['feasible_outcome_summaries']['Kloc+']['count'], 0)

    def test_invalid_budgets_and_outcomes(self):
        for budget in [-1, float('nan'), float('inf'), -float('inf'), True, None, '100', [], 10**1000]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                choose([row(1, 1)], budget)
        for field in ['Effort-', 'Kloc+']:
            for value in [-1, float('nan'), float('inf'), True, '3', []]:
                bad = row(1, 1)
                bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    choose([bad], 100)
        with self.assertRaises(ValueError):
            choose([None], 100)

    def test_no_mutation_no_model_fit_and_no_invented_configuration(self):
        rows = [row(5, 10, ' h '), row(20, 30, 'l')]
        rows[0]['team'] = ' ? '
        before = copy.deepcopy(rows)
        with patch('predict_effort.fit_training', side_effect=AssertionError('Unexpected fitting')):
            report = choose(iter(rows), 10)
        self.assertEqual(rows, before)
        self.assertEqual(report['selected_project']['characteristics'],
                         dict.fromkeys(PREDICTORS, 'h') | {'team': None})
        report['selected_project']['characteristics']['prec'] = 'changed'
        self.assertEqual(rows, before)

    def test_json_explanation_and_caveats(self):
        report = choose([row(3, 10)], 100)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(render_text(report), report['readable_explanation'])
        for text in ['not a monetary budget', 'sponsor-approved', 'not demonstrated business value',
                     'More code is not necessarily better', 'do not prove', 'no objective-evaluation budget']:
            self.assertIn(text, report['readable_explanation'])

    def test_csv_missing_tokens_and_cli_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            temp = Path(directory)
            dataset = temp / 'fixture.csv'
            with dataset.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows([row('?', 100), row(100, 'NA'), row(100, 20)])
            output = temp / 'selection.json'
            cmd = [sys.executable, str(ROOT / 'source/choose.py'), str(dataset), '--budget', '100', '--output', str(output)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['selected_project']['csv_data_row'], 3)
            self.assertEqual(report['counts']['excluded_missing_required_outcomes'], 2)
            self.assertEqual(run.stdout, report['readable_explanation'])
            output.unlink()
            for budget in ['-1', 'nan', 'inf', 'bad']:
                invalid = cmd.copy()
                invalid[invalid.index('--budget') + 1] = budget
                run = subprocess.run(invalid, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2)
                self.assertFalse(output.exists())
            # A valid but infeasible query is a successful explicit result.
            cmd[cmd.index('--budget') + 1] = '0'
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'no_feasible_candidate')

    def test_independent_exhaustive_actual_csv(self):
        with DATASET.open(newline='') as stream:
            raw = list(csv.DictReader(stream))
        # Independent Decimal parsing and pairwise dominance, no production key/sort.
        missing = {'', '?', 'NA', 'N/A', 'null'}
        for budget in [Decimal('0'), Decimal('100'), Decimal('100000')]:
            eligible = []
            feasible = []
            for number, item in enumerate(raw, 1):
                if any(item[field].strip() in missing for field in ['Effort-', 'Kloc+']):
                    continue
                effort, value = Decimal(item['Effort-']), Decimal(item['Kloc+'])
                eligible.append(number)
                if effort <= budget:
                    feasible.append((number, effort, value))
            winners = []
            for candidate in feasible:
                n, e, v = candidate
                dominated = any(v2 > v or (v2 == v and e2 < e) or
                                (v2 == v and e2 == e and n2 < n) for n2, e2, v2 in feasible)
                if not dominated:
                    winners.append(candidate)
            report = choose(read_projects(DATASET), float(budget))
            self.assertEqual(report['counts']['eligible_rows'], len(eligible))
            self.assertEqual(report['counts']['feasible_candidates'], len(feasible))
            if winners:
                self.assertEqual(len(winners), 1)
                selected = report['selected_project']
                self.assertEqual(selected['csv_data_row'], winners[0][0])
                self.assertEqual(selected['observed_outcomes']['Kloc+'], float(winners[0][2]))
                self.assertEqual(selected['observed_outcomes']['Effort-'], float(winners[0][1]))
            else:
                self.assertEqual(report['status'], 'no_feasible_candidate')


if __name__ == '__main__':
    unittest.main(verbosity=2)
