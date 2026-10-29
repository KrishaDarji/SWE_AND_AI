"""Constrained fixtures, CLI checks and independent exhaustive CSV selection."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from choose import choose, choose_rows, render_text
from summarize import CONFIGURATION, OBJECTIVES

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def row(mre=1, acc=2, pred=3, estimators=100):
    return dict(zip([*CONFIGURATION, *OBJECTIVES],
                    [estimators, 'squared_error', 3, 0, 5, mre, acc, pred]))


class ChooseTests(unittest.TestCase):
    def test_feasibility_and_inclusive_boundary(self):
        r = choose_rows([row(0, estimators=101), row(2, estimators=99), row(1)])
        self.assertEqual(r['selected_record']['row_id'], 3)
        self.assertEqual(r['cap_evidence'], {'recorded_N_estimators': 100, 'cap': 100,
                                           'satisfies_inclusive_cap': True})
        self.assertEqual(r['counts'], {'candidate': 3, 'complete': 3, 'excluded': 1,
                                      'feasible': 2, 'incomplete': 0, 'complete_above_cap': 1})
        self.assertEqual(r['tradeoffs']['unconstrained_complete_winner']['row_id'], 1)
        self.assertFalse(r['tradeoffs']['unconstrained_winner_satisfies_cap'])

    def test_mre_direction_and_secondary_tradeoff(self):
        r = choose_rows([row(1, -50, -50), row(2, 100, 100)])
        self.assertEqual(r['selected_record']['row_id'], 1)
        self.assertEqual(r['objective_directions'], dict(OBJECTIVES))
        for n in ['ACC+', 'PRED40+']:
            self.assertEqual(r['tradeoffs']['feasible_objective_champions'][n]['selected_difference']['improvement_in_objective_direction'], -150)

    def test_all_tie_levels_deterministic_and_original_row_ids(self):
        rows = [row(0, estimators=101), row(2, 100, 100), row(1, 4, 100),
                row(1, 5, 2), row(1, 5, 3), row(1, 5, 3)]
        r = choose_rows(rows)
        self.assertEqual(r['selected_record']['row_id'], 5)
        self.assertEqual(r['selected_record']['csv_line'], 6)
        self.assertEqual(r['tie_evidence'], {'minimum_mre_count': 4, 'after_acc_count': 3,
                         'after_pred40_count': 2, 'final_tied_row_ids': [5, 6]})
        self.assertEqual(r, choose_rows(rows))

    def test_every_missing_field_excluded(self):
        rows = []
        for field in [*CONFIGURATION, *OBJECTIVES]:
            for missing in [None, '', '  ']:
                r = row(mre=-100)
                r[field] = missing
                rows.append(r)
        rows.append(row())
        r = choose_rows(rows)
        self.assertEqual(r['counts']['incomplete'], 24)
        self.assertEqual(r['counts']['feasible'], 1)
        self.assertEqual(r['selected_record']['row_id'], 25)
        self.assertEqual(len(r['excluded_records']), 24)
        r = choose_rows([{'MRE-': 0}, row(estimators=101)])
        self.assertEqual(r['counts']['excluded'], 2)
        self.assertEqual(r['status'], 'no_feasible_candidate')

    def test_no_feasible_candidates_and_empty_input(self):
        for rows in [[], (), [row(estimators=101)], [row(mre=None)], [{}]]:
            with self.subTest(rows=rows):
                r = choose_rows(rows)
                self.assertEqual(r['status'], 'no_feasible_candidate')
                self.assertIsNone(r['selected_record'])
                self.assertIsNone(r['cap_evidence'])
                self.assertIsNone(r['tradeoffs'])
                self.assertIn('No feasible candidate', render_text(r))
                self.assertEqual(r['counts']['candidate'], r['counts']['excluded'])

    def test_invalid_caps(self):
        for cap in [0, -1, float('nan'), float('inf'), -float('inf'), True, None,
                    '100', [], 10**1000]:
            with self.subTest(cap=repr(cap)), self.assertRaises(ValueError):
                choose_rows([row()], cap)
        self.assertEqual(choose_rows([row(estimators=99.5)], 99.5)['status'], 'selected')

    def test_invalid_records(self):
        for rows in [None, {}, 'rows', [None], [dict(row(), extra=1)],
                     [row(mre=float('nan'))], [row(acc=True)], [row(pred='bad')],
                     [row(estimators=float('inf'))], [dict(row(), criterion=2)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                choose_rows(rows)

    def test_normalization_detachment_and_scope(self):
        rows = [row(mre=' 1 ', estimators=' 100 ')]
        original = copy.deepcopy(rows)
        r = choose_rows(rows)
        self.assertEqual(r['selected_record']['configuration']['N_estimators'], 100)
        r['selected_record']['configuration']['Max_depth'] = 999
        self.assertEqual(rows, original)
        text = render_text(r)
        for phrase in ['real compute-budget guarantee is unavailable', 'hardware',
                       'Selection optimism', 'independent', 'not a budgeted search']:
            self.assertIn(phrase, text)
        self.assertFalse(r['constraint']['measured_compute_budget'])
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)

    def test_actual_dataset_independent_exhaustive_rule(self):
        with DATASET.open(newline='', encoding='utf-8-sig') as f:
            raw = list(csv.DictReader(f))
        for cap in [1, 40, 100, 1000]:
            with self.subTest(cap=cap):
                report = choose(DATASET, cap)
                complete = [(i, r) for i, r in enumerate(raw, 1)
                            if all(r[n].strip() for n in [*CONFIGURATION, *OBJECTIVES])]
                feasible = [(i, r) for i, r in complete if float(r['N_estimators']) <= cap]
                self.assertEqual(report['counts']['candidate'], len(raw))
                self.assertEqual(report['counts']['complete'], len(complete))
                self.assertEqual(report['counts']['feasible'], len(feasible))
                self.assertEqual(report['counts']['excluded'], len(raw) - len(feasible))
                if not feasible:
                    self.assertIsNone(report['selected_record'])
                    continue
                # Exhaustively sort independent raw CSV values, without application helpers.
                expected = sorted(feasible, key=lambda item: (float(item[1]['MRE-']),
                    -float(item[1]['ACC+']), -float(item[1]['PRED40+']), item[0]))[0]
                self.assertEqual(report['selected_record']['row_id'], expected[0])
                for n in OBJECTIVES:
                    self.assertEqual(report['selected_record']['observed_objectives'][n], float(expected[1][n]))
                for n in CONFIGURATION:
                    value = expected[1][n] if n == 'criterion' else float(expected[1][n])
                    self.assertEqual(report['selected_record']['configuration'][n], value)
                self.assertLessEqual(report['cap_evidence']['recorded_N_estimators'], cap)

    def test_cli_success_no_feasible_invalid_caps_and_bad_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            p = Path(tmp)
            dataset, output = p / 'data.csv', p / 'selection.json'
            with dataset.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[*CONFIGURATION, *OBJECTIVES])
                writer.writeheader()
                writer.writerows([row(estimators=100)])
            cmd = [sys.executable, str(ROOT / 'source/choose.py'), str(dataset), '--output', str(output)]
            good = subprocess.run(cmd + ['--max-estimators', '100'], text=True, capture_output=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(good.stdout, render_text(json.loads(output.read_text())))
            empty = subprocess.run(cmd + ['--max-estimators', '99'], text=True, capture_output=True)
            self.assertEqual(empty.returncode, 0, empty.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'no_feasible_candidate')
            for cap in ['0', '-1', 'nan', 'inf', 'invalid']:
                bad = subprocess.run(cmd + ['--max-estimators', cap], text=True, capture_output=True)
                self.assertEqual(bad.returncode, 2)
            for text in ['wrong\n1\n', ','.join([*CONFIGURATION, *OBJECTIVES]) + '\n',
                         ','.join([*CONFIGURATION, *OBJECTIVES]) + '\n1\n']:
                dataset.write_text(text)
                bad = subprocess.run(cmd, text=True, capture_output=True)
                self.assertEqual(bad.returncode, 2)
            dataset.unlink()
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
