"""Independent exhaustive oracle, deterministic tie fixtures, leakage and CLI tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from fix import complete_proposal, illustrative_proposal, recommend
from predict_effort import estimate, evaluate, normalize_proposal
from summarize import PREDICTORS, read_projects

DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


def configuration(**updates):
    return dict(dict.fromkeys(PREDICTORS, 'n'), **updates)


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_projects(DATASET)
        cls.model, cls.evaluation = evaluate(cls.rows)
        cls.proposal, cls.number = illustrative_proposal(cls.rows)
        cls.result = recommend(cls.rows, cls.proposal)

    def test_independent_exhaustive_minimum_and_changes(self):
        # Deliberately enumerate every training row, without production deduplication.
        original = estimate(self.model, self.proposal)
        possibilities = []
        for i in self.evaluation['split']['training_indices']:
            c = {p: self.rows[i][p].strip() for p in PREDICTORS}
            predicted = estimate(self.model, c)
            distance = sum(self.proposal[p] != c[p] for p in PREDICTORS)
            if predicted < original:
                possibilities.append((distance, predicted, i + 1, c))
        expected = sorted(possibilities, key=lambda c: c[:3])[0]
        r = self.result
        winner = r['recommended_candidate']
        self.assertEqual((winner['change_cost'], winner['predicted_effort'], winner['csv_data_row']), expected[:3])
        self.assertEqual(winner['predictors'], expected[3])
        changed = [{'field': p, 'old': self.proposal[p], 'new': expected[3][p]}
                   for p in PREDICTORS if self.proposal[p] != expected[3][p]]
        self.assertEqual(r['changed_fields'], changed)
        self.assertEqual(r['change_cost'], len(changed))
        self.assertEqual(r['predicted_reduction'], original - expected[1])
        self.assertGreater(r['predicted_reduction'], 0)
        self.assertEqual(r['original_predicted_effort'], original)
        self.assertEqual(r['search']['training_rows'], len(self.evaluation['split']['training_indices']))
        keys = {tuple(normalize_proposal({p: self.rows[i][p] for p in PREDICTORS}))
                for i in self.evaluation['split']['training_indices']}
        self.assertEqual(r['search']['unique_candidates_evaluated'], len(keys))
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)

    def test_demo_earliest_training_predictors_and_no_mutation(self):
        index = min(self.evaluation['split']['training_indices'])
        self.assertEqual(self.number, index + 1)
        self.assertEqual(self.proposal, {p: self.rows[index][p] for p in PREDICTORS})
        before_rows, before_proposal = copy.deepcopy(self.rows), copy.deepcopy(self.proposal)
        recommend(self.rows, self.proposal)
        self.assertEqual(before_rows, self.rows)
        self.assertEqual(before_proposal, self.proposal)

    def test_heldout_outcomes_do_not_change_entire_recommendation(self):
        rows = copy.deepcopy(self.rows)
        for i in self.evaluation['split']['evaluation_indices']:
            rows[i]['Effort-'] = rows[i]['Effort-'] * 100000 + 9999
        self.assertEqual(illustrative_proposal(rows), (self.proposal, self.number))
        self.assertEqual(recommend(rows, self.proposal), self.result)

    def controlled(self, rows, scores, train=None):
        if train is None:
            train = list(range(len(rows)))
        fake = {'split': {'training_indices': train, 'evaluation_samples': len(rows) - len(train)},
                'dependency_versions': {}}
        def prediction(model, proposal):
            return scores[tuple(normalize_proposal(proposal))]
        with patch('fix.evaluate', return_value=(object(), fake)), patch('fix.estimate', side_effect=prediction):
            return recommend(rows, configuration())

    def test_cost_first_then_prediction_then_csv_order_and_duplicates(self):
        original = configuration()
        a, b = configuration(prec='a'), configuration(flex='b')
        far = configuration(prec='x', flex='x')
        duplicate = dict(a, prec=' a ')
        rows = [b, a, far, duplicate]
        key = lambda c: tuple(normalize_proposal(c))
        scores = {key(original): 100, key(a): 40, key(b): 50, key(far): 1}
        r = self.controlled(rows, scores, [3, 2, 1, 0])
        self.assertEqual(r['recommended_candidate']['csv_data_row'], 2)
        self.assertEqual(r['change_cost'], 1)
        self.assertEqual(r['search']['duplicate_complete_rows'], 1)
        self.assertEqual(r['search']['unique_candidates_evaluated'], 3)
        scores[key(b)] = 40
        r = self.controlled(rows, scores, [3, 2, 1, 0])
        self.assertEqual(r['recommended_candidate']['csv_data_row'], 1)
        self.assertEqual(r['changed_fields'], [{'field': 'flex', 'old': 'n', 'new': 'b'}])

    def test_strict_comparison_no_improvement_and_empty_candidate_pool(self):
        original, equal, worse = configuration(), configuration(prec='e'), configuration(prec='w')
        key = lambda c: tuple(normalize_proposal(c))
        scores = {key(original): 10, key(equal): 10, key(worse): 20}
        for rows in [[], [original, equal, worse], [configuration(prec=None)]]:
            r = self.controlled(rows, scores)
            self.assertEqual(r['status'], 'no_improving_candidate')
            self.assertIsNone(r['recommended_candidate'])
            self.assertIsNone(r['change_cost'])
            self.assertIsNone(r['predicted_reduction'])
            self.assertEqual(r['changed_fields'], [])
            self.assertIsNotNone(r['no_improvement_reason'])
        scores[key(equal)] = 10 - 1e-12
        self.assertEqual(self.controlled([equal], scores)['status'], 'recommendation')

    def test_only_complete_training_configurations_are_candidates(self):
        original, heldout = configuration(), configuration(prec='best')
        incomplete = configuration(flex='?')
        scores = {tuple(normalize_proposal(original)): 100}
        r = self.controlled([original, heldout, incomplete], scores, [0, 2])
        self.assertEqual(r['status'], 'no_improving_candidate')
        self.assertEqual(r['search']['excluded_incomplete_training_rows'], 1)
        self.assertEqual(r['search']['unique_candidates_evaluated'], 1)
        self.assertEqual(r['search']['held_out_rows_excluded'], 1)

    def test_complete_normalization_and_general_missing_api_preserved(self):
        proposal = {p: ' n\t' for p in PREDICTORS}
        self.assertEqual(complete_proposal(proposal), configuration())
        self.assertEqual(normalize_proposal({}), [None] * len(PREDICTORS))
        unseen = configuration(prec='new_category')
        self.assertEqual(complete_proposal(unseen), unseen)
        self.assertEqual(recommend(self.rows, {p: ' ' + v + ' ' for p, v in self.proposal.items()})['recommended_candidate'],
                         self.result['recommended_candidate'])

    def test_invalid_incomplete_rejected_before_fitting(self):
        invalid = [None, [], 1, {}, {'prec': 'n'}]
        invalid += [configuration(prec=v) for v in [None, '', '?', 'NA', 'N/A', 'null', ' ? ', 1, True, [], {}]]
        invalid += [dict(configuration(), **{p: 'n'}) for p in ['Effort-', 'Kloc+', 'Defects-', 'idX', 'futureX', 'typo']]
        for proposal in invalid:
            with self.subTest(proposal=proposal), patch('fix.evaluate') as fitting:
                with self.assertRaises(ValueError):
                    recommend(self.rows, proposal)
                fitting.assert_not_called()

    def test_cli_proposal_demo_and_invalid_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            tmp = Path(tmp)
            output, proposal = tmp / 'repair.json', tmp / 'proposal.json'
            cmd = [sys.executable, str(ROOT / 'source/fix.py'), str(DATASET), '--demo',
                   '--example-proposal', str(proposal), '--output', str(output)]
            p = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(proposal.read_text()), self.proposal)
            demo = json.loads(output.read_text())
            self.assertEqual(demo['demo']['proposal_csv_data_row'], self.number)
            cmd = cmd[:3] + ['--proposal', str(proposal), '--output', str(output)]
            p = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)['recommended_candidate'], self.result['recommended_candidate'])
            for text in ['{"prec":"n","prec":"h"}', '{bad', '[]', '{"Effort-":1}', '{}', '{"prec":null}']:
                proposal.write_text(text)
                output.unlink(missing_ok=True)
                p = subprocess.run(cmd, text=True, capture_output=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
