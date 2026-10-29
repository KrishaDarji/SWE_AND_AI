"""Independent arithmetic fixtures and full-dataset consistency for Step 6."""
import copy
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from justify import justify, justify_rows, render_text
from summarize import CONFIGURATION, OBJECTIVES

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def row(mre=1, acc=2, pred=3, x=1, criterion='a'):
    return dict(zip(CONFIGURATION, [x, criterion, 1, 0, 5]),
                **{'MRE-': mre, 'ACC+': acc, 'PRED40+': pred})


class JustifyTests(unittest.TestCase):
    def test_mre_first_even_if_secondary_objectives_are_much_worse(self):
        report = justify_rows([row(1, 1, 1), row(2, 100, 100)])
        self.assertEqual(report['recommended_record']['row_id'], 1)
        self.assertEqual(report['objective_directions'],
                         {'MRE-': 'minimize', 'ACC+': 'maximize', 'PRED40+': 'maximize'})
        for name in ('ACC+', 'PRED40+'):
            p = report['historical_positions'][name]
            self.assertEqual(p['rank_from_best_competition'], 2)
            self.assertEqual(p['references']['historical_best']['improvement_in_objective_direction'], -99)
        self.assertTrue(report['tradeoffs']['nondominated_among_complete_records'])

    def test_acc_tie_then_pred_tie_then_earliest_row(self):
        rows = [row(2, 100, 100), row(1, 5, 100), row(1, 6, 8),
                row(1, 6, 9), row(1, 6, 9)]
        r = justify_rows(rows)
        self.assertEqual(r['recommended_record']['row_id'], 4)
        self.assertEqual(r['recommended_record']['csv_line'], 5)
        s = r['selection']
        self.assertEqual([s['minimum_mre_tie_count'], s['remaining_after_acc_tie_count'],
                          s['remaining_after_pred40_tie_count']], [4, 3, 2])
        self.assertEqual(s['final_tied_row_ids'], [4, 5])

    def test_missing_each_objective_excluded_and_reported(self):
        rows = [row(None, 9, 9), row(-100, None, 9), row(-100, 9, None), row(4, 4, 4)]
        r = justify_rows(rows)
        self.assertEqual(r['recommended_record']['row_id'], 4)
        self.assertEqual(r['selection']['eligible_count'], 1)
        self.assertEqual(r['selection']['excluded_records'], [
            {'row_id': 1, 'missing_objectives': ['MRE-']},
            {'row_id': 2, 'missing_objectives': ['ACC+']},
            {'row_id': 3, 'missing_objectives': ['PRED40+']}])
        self.assertEqual(r['historical_positions']['MRE-']['rank_from_best_competition'], 3)
        self.assertEqual(r['historical_positions']['MRE-']['summary']['missing'], 1)
        self.assertEqual(r['reference_records']['MRE-_champion']['row_id'], 4)

    def test_missing_all_objectives_no_eligible_record(self):
        for rows in ([row(None, 1, 1)], [{}], [row(1, '', None)]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'No record'):
                justify_rows(rows)

    def test_hand_calculated_distribution_and_reference_differences(self):
        r = justify_rows([row(2, 10, 3), row(4, 30, 5), row(6, 20, 7)])
        p = r['historical_positions']['MRE-']
        self.assertEqual(p['summary'], {'n': 3, 'missing': 0, 'mean': 4, 'std_sample': 2,
                                       'median': 4, 'q25': 3, 'q75': 5, 'min': 2, 'max': 6})
        self.assertEqual([p['strictly_below_count'], p['equal_count'], p['strictly_above_count']], [0, 1, 2])
        self.assertAlmostEqual(p['lower_tail_midrank_percentile'], 100 / 6)
        self.assertEqual(p['references']['historical_mean'], {'value': 4,
            'selected_minus_reference': -2, 'improvement_in_objective_direction': 2,
            'observed_relation': 'better'})
        acc = r['historical_positions']['ACC+']
        self.assertEqual(acc['rank_from_best_competition'], 3)
        self.assertEqual(acc['references']['historical_median']['selected_minus_reference'], -10)
        ref = r['reference_records']['ACC+_champion']
        self.assertEqual(ref['row_id'], 2)
        self.assertEqual(ref['differences']['MRE-']['improvement_in_objective_direction'], 2)
        self.assertEqual(ref['differences']['ACC+']['improvement_in_objective_direction'], -20)

    def test_champion_ties_use_earliest_not_recommendation_tie_rule(self):
        r = justify_rows([row(1, 1, 3), row(1, 2, 3)])
        self.assertEqual(r['recommended_record']['row_id'], 2)
        self.assertEqual(r['reference_records']['MRE-_champion']['row_id'], 1)
        self.assertEqual(r['reference_records']['earliest_complete_record']['row_id'], 1)
        self.assertEqual(r['historical_positions']['MRE-']['equal_count'], 2)
        self.assertEqual(r['historical_positions']['MRE-']['lower_tail_midrank_percentile'], 50)

    def test_same_configuration_and_setting_evidence_not_individual_effects(self):
        rows = [row(1, x=1), row(3, x=1), row(5, x=2), row(7, x=2)]
        r = justify_rows(rows)
        setting = r['setting_evidence']['N_estimators']
        # Overall mean=4; SS_between=4*(2**2)=16; SS_total=9+1+1+9=20.
        self.assertAlmostEqual(setting['eta_squared'], .8)
        self.assertEqual(setting['selected_group']['mre']['mean'], 2)
        self.assertEqual(setting['selected_group']['mean_minus_included_mean'], -2)
        self.assertEqual(setting['selected_group']['mre']['n'], 2)
        self.assertTrue(setting['selected_group']['sparse'])
        self.assertEqual(r['same_configuration_evidence']['row_ids'], [1, 2])
        self.assertEqual(r['same_configuration_evidence']['objectives']['MRE-']['mean'], 2)
        self.assertIsNone(r['setting_evidence']['criterion']['eta_squared'])
        self.assertIsNone(r['evidence_scope']['predictions'])
        text = render_text(r)
        self.assertIn('not causal', text)
        self.assertIn('prediction contributions', text)
        self.assertIn('Selection optimism', text)
        self.assertIn('independent', text)

    def test_quartile_boundary_selected_group(self):
        rows = [row(i, 1, 1, x=i) for i in range(1, 10)]
        r = justify_rows(rows)
        setting = r['setting_evidence']['N_estimators']
        self.assertEqual(setting['grouping']['cuts'], [3, 5, 7])
        self.assertEqual(setting['selected_group']['mre']['n'], 3)
        self.assertEqual(setting['selected_group']['mre']['mean'], 2)

    def test_missing_configuration_retained_and_explicit(self):
        r = justify_rows([{'MRE-': '1', 'ACC+': '2', 'PRED40+': '3'}])
        self.assertEqual(r['recommended_record']['missing_configuration_fields'], list(CONFIGURATION))
        for setting in r['setting_evidence'].values():
            self.assertIsNone(setting['selected_group'])
            self.assertEqual(setting['missing_predictor_mre']['n'], 1)
        self.assertIn('Missing selected setting', render_text(r))

    def test_invalid_reusable_inputs(self):
        for bad in (None, [], {}, 'rows', [None], [dict(row(), unknown=1)],
                    [row(float('nan'))], [row(acc=float('inf'))], [row(pred=True)],
                    [row(x=False)], [row(criterion=3)], [row(mre='invalid')]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                justify_rows(bad)

    def test_inputs_not_mutated_and_results_detached(self):
        rows = [row(1), row(2)]
        before = copy.deepcopy(rows)
        r = justify_rows(rows)
        r['recommended_record']['configuration']['Max_depth'] = 999
        self.assertEqual(rows, before)
        self.assertEqual(justify_rows(rows)['recommended_record']['configuration']['Max_depth'], 5)

    def test_constant_singleton_and_negative_scales_allowed(self):
        r = justify_rows([row(-2, -3, -4)])
        for name, p in r['historical_positions'].items():
            self.assertEqual(p['rank_from_best_competition'], 1)
            self.assertEqual(p['lower_tail_midrank_percentile'], 50)
            self.assertIsNone(p['summary']['std_sample'])
        self.assertIn('not a universally best', r['preference'])
        self.assertIn('not proof of perfect', render_text(r))

    def test_cli_outputs_and_invalid_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            path = Path(tmp)
            data = path / 'fixture.csv'
            with data.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[*CONFIGURATION, *OBJECTIVES])
                writer.writeheader()
                writer.writerows([row(2), row(1)])
            cmd = [sys.executable, str(ROOT / 'source/justify.py'), str(data), '--output-dir', str(path / 'out')]
            good = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            report = json.loads((path / 'out/justification.json').read_text())
            self.assertEqual(report['recommended_record']['row_id'], 2)
            self.assertEqual((path / 'out/justification.txt').read_text(), render_text(report))
            self.assertEqual(good.stdout, render_text(report))
            for content in ('wrong\n1\n', ','.join([*CONFIGURATION, *OBJECTIVES]) + '\n',
                            ','.join([*CONFIGURATION, *OBJECTIVES]) + '\n1\n'):
                data.write_text(content)
                result = subprocess.run(cmd, text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Error:', result.stderr)
            data.unlink()
            result = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_actual_dataset_independently_computed_evidence(self):
        report = justify(DATASET)
        with DATASET.open(newline='') as f:
            raw = list(csv.DictReader(f))
        complete = [(i, r) for i, r in enumerate(raw, 1) if all(r[n].strip() for n in OBJECTIVES)]
        index, selected = min(complete, key=lambda item: (float(item[1]['MRE-']),
            -float(item[1]['ACC+']), -float(item[1]['PRED40+']), item[0]))
        self.assertEqual(report['recommended_record']['row_id'], index)
        self.assertEqual(report['selection']['excluded_count'], len(raw) - len(complete))
        for name, direction in OBJECTIVES.items():
            values = [float(r[name]) for r in raw if r[name].strip()]
            chosen = float(selected[name])
            p = report['historical_positions'][name]
            self.assertAlmostEqual(p['summary']['mean'], statistics.mean(values))
            self.assertEqual(p['summary']['median'], statistics.median(values))
            better = sum(v < chosen if direction == 'minimize' else v > chosen for v in values)
            self.assertEqual(p['rank_from_best_competition'], 1 + better)
            self.assertEqual(p['references']['historical_mean']['selected_minus_reference'], chosen - statistics.mean(values))
        for name, s in report['setting_evidence'].items():
            group = s['selected_group']
            if s['grouping']['kind'] in ('categories', 'exact_values'):
                expected = [float(r['MRE-']) for r in raw if r['MRE-'].strip() and
                            (r[name] if name == 'criterion' else float(r[name])) == s['selected_value']]
            else:
                d = group['definition']
                expected = [float(r['MRE-']) for r in raw if r['MRE-'].strip() and
                            (float(r[name]) >= d['lower'] if d['lower_inclusive'] else float(r[name]) > d['lower']) and
                            float(r[name]) <= d['upper']]
            self.assertEqual(group['mre']['n'], len(expected))
            self.assertEqual(group['mre']['mean'], statistics.mean(expected))
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertIn(str(index), render_text(report))


if __name__ == '__main__':
    unittest.main(verbosity=2)
