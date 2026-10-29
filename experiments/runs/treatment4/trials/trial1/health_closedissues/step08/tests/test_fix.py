"""Recorded repair tests, with independent exhaustive and CSV calculations."""
import copy
import csv
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from fix import repair, repair_rows, render_text
from summarize import CONFIGURATION, OBJECTIVES

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def row(mre=1, acc=2, pred=3, **settings):
    result = dict(zip(CONFIGURATION, [40, 'squared_error', 3, 2.75, 10]))
    result.update(settings)
    return {**result, 'MRE-': mre, 'ACC+': acc, 'PRED40+': pred}


class RepairTests(unittest.TestCase):
    def test_cost_precedes_error_and_numeric_magnitude_counts_once(self):
        rows = [row(10), row(0, N_estimators=41, Max_depth=11),
                row(9, N_estimators=100000)]
        r = repair_rows(rows, 1)
        self.assertEqual(r['recommendation']['row_id'], 3)
        self.assertEqual(r['change_cost'], 1)
        self.assertEqual(r['observed_MRE_reduction'], 1)
        self.assertEqual(r['changed_fields'], [{'field': 'N_estimators', 'old': 40, 'new': 100000}])

    def test_mre_then_earliest_ties_ignore_secondary_objectives(self):
        rows = [row(10), row(8, criterion='z'), row(7, acc=-10, pred=-20, criterion='b'),
                row(7, acc=100, pred=100, criterion='a')]
        r = repair_rows(rows, 1)
        self.assertEqual(r['recommendation']['row_id'], 3)
        self.assertEqual(r['tie_evidence']['final_tied_row_ids'], [3, 4])
        self.assertEqual(r['secondary_differences']['ACC+']['selected_minus_reference'], -12)
        self.assertEqual(r['secondary_differences']['PRED40+']['observed_relation'], 'worse')

    def test_duplicates_zero_cost_and_normalization(self):
        baseline = row(10, N_estimators=' 40.0 ', criterion=' squared_error ')
        rows = [baseline, row(8), row(7), row(7)]
        r = repair_rows(rows, 1)
        self.assertEqual(r['change_cost'], 0)
        self.assertEqual(r['changed_fields'], [])
        self.assertFalse(r['setting_change_identified'])
        self.assertIn('no setting change was identified', r['setting_change_statement'])
        self.assertEqual(r['recommendation']['row_id'], 3)
        self.assertEqual(r['tie_evidence']['final_tied_row_ids'], [3, 4])

    def test_strict_improvement_and_explicit_empty_result(self):
        for rows in ([row()], [row(), row(), row(2)], [row(-2), row(-2), row(-1)]):
            with self.subTest(rows=rows):
                r = repair_rows(rows, 1)
                self.assertEqual(r['status'], 'no_improving_candidate')
                self.assertIsNone(r['recommendation'])
                self.assertIsNone(r['change_cost'])
                self.assertIsNone(r['observed_MRE_reduction'])
                self.assertIsNone(r['secondary_differences'])
                self.assertEqual(r['counts']['strictly_improving'], 0)

    def test_missing_outcomes_settings_and_overlapping_exclusions(self):
        rows = [row(10, acc=None), row(5, pred=None, Max_depth=11),
                row(None), row(0, criterion=None), row(None, Max_depth=None)]
        r = repair_rows(rows, 1)
        self.assertEqual(r['recommendation']['row_id'], 2)
        self.assertEqual(r['counts'], {'recorded_rows': 5, 'complete_settings_and_mre': 2,
            'excluded': 3, 'missing_settings': 2, 'missing_mre': 2,
            'strictly_improving': 1, 'complete_not_improving': 1})
        for name in ('ACC+', 'PRED40+'):
            self.assertIsNone(r['secondary_differences'][name]['selected_minus_reference'])
            self.assertEqual(r['secondary_differences'][name]['observed_relation'], 'unavailable')
        self.assertIsNone(r['baseline']['observed_objectives']['ACC+'])
        self.assertIsNone(r['recommendation']['observed_objectives']['PRED40+'])

    def test_each_missing_setting_and_mre_rejects_baseline_and_excludes_candidate(self):
        for name in (*CONFIGURATION, 'MRE-'):
            for blank in (None, '', '  '):
                missing = row(0)
                missing[name] = blank
                with self.subTest(name=name, blank=blank):
                    with self.assertRaises(ValueError):
                        repair_rows([missing, row()], 1)
                    r = repair_rows([row(), missing], 1)
                    self.assertEqual(r['counts']['excluded'], 1)
                    self.assertIsNone(r['recommendation'])

    def test_invalid_baseline_rows_and_reusable_inputs(self):
        for baseline in (0, -1, 2, True, False, 1.0, '1', None):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                repair_rows([row()], baseline)
        for rows in ([], None, {}, [None], [dict(row(), unknown=1)],
                     [row(float('nan'))], [row(N_estimators=True)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                repair_rows(rows, 1)

    def test_nonfirst_baseline_inputs_detached_and_exact_changed_fields(self):
        rows = [row(0, criterion='absolute_error', N_estimators=200, Max_depth=15,
                    Min_sample_leaves=1, Min_impurity_decrease=0), row(10)]
        before = copy.deepcopy(rows)
        r = repair_rows(rows, 2)
        self.assertEqual(rows, before)
        self.assertEqual(r['baseline']['row_id'], 2)
        self.assertEqual(r['change_cost'], 5)
        expected = [{'field': n, 'old': rows[1][n], 'new': rows[0][n]} for n in CONFIGURATION]
        self.assertEqual(r['changed_fields'], expected)
        r['recommendation']['configuration']['criterion'] = 'mutated'
        self.assertEqual(rows, before)
        json.loads(render_text(r))

    def test_independent_exhaustive_generated_grid(self):
        rng = random.Random(8041)
        rows = []
        for values in itertools.product((40, 200), ('a', 'b'), (1, 3), (0, 2.75), (5, 10)):
            rows.append({**dict(zip(CONFIGURATION, values)),
                         'MRE-': rng.randrange(5), 'ACC+': None, 'PRED40+': rng.random()})
        rows.extend(copy.deepcopy(rows[:5]))
        for baseline_id, baseline in enumerate(rows, 1):
            # Independently enumerate by cost layers, then error, then original position.
            eligible = [(i, candidate) for i, candidate in enumerate(rows, 1)
                        if candidate['MRE-'] < baseline['MRE-']]
            expected = None
            for cost in range(6):
                layer = [(c['MRE-'], i) for i, c in eligible
                         if len([n for n in CONFIGURATION if c[n] != baseline[n]]) == cost]
                if layer:
                    expected = (cost, *sorted(layer)[0])
                    break
            r = repair_rows(rows, baseline_id)
            if expected is None:
                self.assertIsNone(r['recommendation'])
            else:
                self.assertEqual((r['change_cost'], r['recommendation']['observed_objectives']['MRE-'],
                                  r['recommendation']['row_id']), expected)

    def test_actual_csv_independent_exhaustive_calculation(self):
        with DATASET.open(newline='') as handle:
            raw = list(csv.DictReader(handle))
        def normalize(r):
            return {n: (None if not v.strip() else v.strip() if n == 'criterion' else float(v))
                    for n, v in r.items()}
        rows = [normalize(r) for r in raw]
        baseline = rows[0]
        eligible = [(sum(c[n] != baseline[n] for n in CONFIGURATION), c['MRE-'], i)
                    for i, c in enumerate(rows, 1)
                    if all(c[n] is not None for n in (*CONFIGURATION, 'MRE-'))
                    and c['MRE-'] < baseline['MRE-']]
        expected = sorted(eligible)[0]
        r = repair(DATASET, 1)
        self.assertEqual((r['change_cost'], r['recommendation']['observed_objectives']['MRE-'],
                          r['recommendation']['row_id']), expected)
        selected = rows[expected[2] - 1]
        self.assertEqual(r['recommendation']['configuration'], {n: selected[n] for n in CONFIGURATION})
        self.assertEqual(r['changed_fields'], [{'field': n, 'old': baseline[n], 'new': selected[n]}
            for n in CONFIGURATION if baseline[n] != selected[n]])
        self.assertEqual(r['observed_MRE_reduction'], baseline['MRE-'] - selected['MRE-'])
        for n in ('ACC+', 'PRED40+'):
            self.assertEqual(r['secondary_differences'][n]['selected_minus_reference'], selected[n] - baseline[n])
        self.assertEqual(r['counts']['strictly_improving'], len(eligible))

    def test_cli_real_data_and_invalid_baseline_and_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output = Path(tmp) / 'repair.json'
            command = [sys.executable, '-B', str(ROOT / 'source/fix.py'), str(DATASET),
                       '--baseline-row', '1', '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), json.loads(result.stdout))
            for value in ('0', '-1', '9999999', '1.5'):
                output.unlink(missing_ok=True)
                invalid = command.copy()
                invalid[5] = value
                result = subprocess.run(invalid, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())
            bad = Path(tmp) / 'bad.csv'
            bad.write_text('wrong,header\n1,2\n')
            invalid = command.copy()
            invalid[3] = str(bad)
            self.assertEqual(subprocess.run(invalid, capture_output=True).returncode, 2)


if __name__ == '__main__':
    unittest.main()
