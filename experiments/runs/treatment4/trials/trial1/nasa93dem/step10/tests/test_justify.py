"""Independent evidence arithmetic, leakage isolation, contracts and CLI checks."""
import copy
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
from justify import justify, render_text
from predict_effort import estimate, evaluate, normalize_proposal
from summarize import PREDICTORS, read_projects

DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


class JustificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_projects(DATASET)
        cls.proposal = json.loads((ROOT / 'example_proposal.json').read_text())
        cls.model, cls.evaluation = evaluate(cls.rows)
        cls.report = justify(cls.rows, cls.proposal, True)

    def test_prediction_partition_and_reproducibility(self):
        r = self.report
        self.assertEqual(r['estimate']['value'], estimate(self.model, self.proposal))
        self.assertEqual(r['evaluation'], self.evaluation)
        self.assertEqual(r, justify(self.rows, self.proposal, True))
        self.assertEqual(r['readable_explanation'], render_text(r))
        self.assertIn('Illustrative', r['proposal']['label'])
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)

    def test_training_context_independent_arithmetic(self):
        r = self.report
        indices = self.evaluation['split']['training_indices']
        self.assertEqual(r['training_context']['row_indices'], indices)
        self.assertFalse(set(indices) & set(self.evaluation['split']['evaluation_indices']))
        efforts = sorted(self.rows[i]['Effort-'] for i in indices)
        s = r['training_context']['effort_summary']
        self.assertEqual(s['count'], len(efforts))
        self.assertAlmostEqual(s['mean'], sum(efforts) / len(efforts))
        self.assertEqual(s['min'], min(efforts))
        self.assertEqual(s['max'], max(efforts))
        values = normalize_proposal(self.proposal)
        candidates = []
        exact = 0
        for i in indices:
            v = normalize_proposal({p: self.rows[i][p] for p in PREDICTORS})
            distance = sum(a != b for a, b in zip(values, v))
            candidates.append((distance, i))
            exact += distance == 0
        expected = sorted(candidates)[:5]
        self.assertEqual([(n['mismatch_count'], n['row_index']) for n in r['training_context']['nearest_examples']], expected)
        self.assertEqual(r['training_context']['exact_configuration_count'], exact)
        for c in r['proposal']['characteristics']:
            observed = [self.rows[i]['Effort-'] for i in indices
                        if normalize_proposal({c['field']: self.rows[i][c['field']]})[PREDICTORS.index(c['field'])] == c['normalized_value']]
            s = c['matching_category_effort_summary']
            self.assertEqual(s['count'], len(observed))
            self.assertEqual(s['sum'], sum(observed) if observed else None)
        for key, column in [('model_mae', 'predicted'), ('baseline_mae', 'baseline_predicted')]:
            heldout = r['evaluation']['held_out_predictions']
            self.assertAlmostEqual(r['evaluation'][key], sum(abs(x['actual'] - x[column]) for x in heldout) / len(heldout))

    def test_all_local_sensitivities_match_direct_model_queries(self):
        r = self.report
        normalized = r['proposal']['normalized']
        indices = r['training_context']['row_indices']
        for f in r['local_model_sensitivity']:
            expected_levels = {self.rows[i][f['field']] for i in indices} - {None, normalized[f['field']]}
            self.assertEqual({a['replacement'] for a in f['alternatives']}, expected_levels)
            for a in f['alternatives']:
                proposal = dict(normalized, **{f['field']: a['replacement']})
                self.assertEqual(a['estimated_effort'], estimate(self.model, proposal))
                self.assertEqual(a['delta_from_proposal'], a['estimated_effort'] - r['estimate']['value'])
                self.assertEqual(a['training_support'], sum(self.rows[i][f['field']] == a['replacement'] for i in indices))

    def test_global_association_independent_training_only_arithmetic(self):
        indices = self.report['training_context']['row_indices']
        for f in self.report['global_historical_associations']['ranked_findings']:
            groups = {}
            for i in indices:
                row = self.rows[i]
                groups.setdefault(row[f['factor']], []).append(row['Effort-'])
            all_values = [v for g in groups.values() for v in g]
            mean = sum(all_values) / len(all_values)
            between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values())
            total = sum((v - mean) ** 2 for v in all_values)
            self.assertAlmostEqual(f['eta_squared'], between / total)

    def test_heldout_target_changes_leave_all_training_explanation_unchanged(self):
        changed = copy.deepcopy(self.rows)
        for i in self.evaluation['split']['evaluation_indices']:
            changed[i]['Effort-'] = changed[i]['Effort-'] * 10000 + 999
        actual = justify(changed, self.proposal, True)
        for key in self.report:
            if key not in ['evaluation', 'readable_explanation']:
                self.assertEqual(actual[key], self.report[key], key)
        self.assertNotEqual(actual['evaluation']['model_mae'], self.report['evaluation']['model_mae'])
        self.assertNotEqual(actual['evaluation']['baseline_mae'], self.report['evaluation']['baseline_mae'])
        # Masking only aggregate evaluation restores identical readable explanation.
        actual['evaluation'] = self.report['evaluation']
        self.assertEqual(render_text(actual), self.report['readable_explanation'])

    def test_missing_unseen_normalization_and_no_mutation(self):
        proposal = {'prec': ' brand_new_category ', 'team': ' ? ', 'rely': None}
        saved = copy.deepcopy(proposal)
        rows_before = copy.deepcopy(self.rows)
        r = justify(self.rows, proposal)
        self.assertEqual(proposal, saved)
        self.assertEqual(self.rows, rows_before)
        self.assertEqual(r['estimate']['value'], estimate(self.model, proposal))
        self.assertTrue(math.isfinite(r['estimate']['value']))
        c = {x['field']: x for x in r['proposal']['characteristics']}
        self.assertEqual(c['prec']['status'], 'unseen')
        self.assertEqual(c['prec']['matching_category_effort_summary']['count'], 0)
        self.assertEqual(c['team']['status'], 'missing')
        self.assertTrue(c['team']['supplied_key'])
        self.assertFalse(c['flex']['supplied_key'])
        self.assertEqual(c['flex']['status'], 'missing')
        missing = justify(self.rows, {})
        self.assertEqual(missing['estimate']['value'], estimate(self.model, {}))
        self.assertTrue(all(c['status'] == 'missing' for c in missing['proposal']['characteristics']))

    def test_invalid_proposals_rejected_before_fitting(self):
        for proposal in [[], None, 1, {'prec': 1}, {'team': True}] + [
                {field: 'n'} for field in ['Effort-', 'Kloc+', 'Defects-', 'idX', 'centerX', 'YearX', 'MonthsX', 'futureX', 'typo']]:
            with self.subTest(proposal=proposal), patch('justify.evaluate') as fit:
                with self.assertRaises(ValueError):
                    justify(self.rows, proposal)
                fit.assert_not_called()

    def test_cli_and_json_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            out = Path(directory)
            cmd = [sys.executable, str(ROOT / 'source/justify.py'), str(DATASET),
                   '--proposal', str(ROOT / 'example_proposal.json'), '--output-dir', str(out), '--illustrative']
            p = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads((out / 'justification.json').read_text())
            self.assertEqual(r['readable_explanation'], p.stdout)
            self.assertEqual((out / 'justification.txt').read_text(), p.stdout)
            self.assertEqual(r['estimate'], self.report['estimate'])
            invalid = out / 'bad.json'
            for text in ['{"prec":"n","prec":"h"}', '{"Effort-":10}', '[]', '{bad']:
                invalid.write_text(text)
                bad_cmd = cmd.copy()
                bad_cmd[bad_cmd.index('--proposal') + 1] = str(invalid)
                bad_cmd[bad_cmd.index('--output-dir') + 1] = str(out / 'invalid-output')
                p = subprocess.run(bad_cmd, capture_output=True, text=True)
                self.assertEqual(p.returncode, 2)
                self.assertFalse((out / 'invalid-output/justification.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
