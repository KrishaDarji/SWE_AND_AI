"""Sequential leakage checks, lifecycle checks, and independent running arithmetic."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from flow import ProposalFlow, replay
from predict_effort import fit_training
from summarize import PREDICTORS


def rows():
    return [dict.fromkeys(PREDICTORS, 'n') | {'Effort-': value}
            for value in [10., 20., 35., None, 50., 80.]]


class FlowTests(unittest.TestCase):
    def test_prediction_recorded_before_target_access(self):
        events = []

        class GuardedRow(dict):
            def get(self, key, default=None):
                if key == 'Effort-':
                    events.append('target')
                return super().get(key, default)

        original = ProposalFlow.receive

        def receive(flow, proposal):
            self.assertEqual(set(proposal), set(PREDICTORS))
            self.assertEqual(events, ['prediction', 'target'] * len(flow._assessments))
            result = original(flow, proposal)
            self.assertIn('predicted_effort', flow._pending)
            events.append('prediction')
            return result

        with patch.object(ProposalFlow, 'receive', receive):
            replay(GuardedRow(row) for row in rows())
        self.assertEqual(events, ['prediction', 'target'] * len(rows()))

    def test_fit_uses_only_previously_revealed_records(self):
        data = rows()
        with patch('flow.fit_training', wraps=fit_training) as fit:
            report = replay(data)
        expected = []
        for i in range(2, len(data)):
            expected.append([{p: row[p] for p in PREDICTORS} | {'Effort-': row['Effort-']}
                             for row in data[:i] if row['Effort-'] is not None])
        # Snapshot at call time: history is mutable, so inspect independent fit inputs below.
        self.assertEqual(fit.call_count, len(expected))
        captured = []

        def capture(history):
            captured.append(copy.deepcopy(history))
            return fit_training(history)

        with patch('flow.fit_training', side_effect=capture):
            self.assertEqual(replay(data), report)
        self.assertEqual(captured, expected)

    def test_missing_outcomes_and_warm_up(self):
        data = [dict.fromkeys(PREDICTORS, 'n') | {'Effort-': x}
                for x in [None, 10., None, 20., None, 30.]]
        report = replay(data)
        items = report['assessments']
        self.assertEqual([r['history_size_before'] for r in items], [0, 0, 1, 1, 2, 2])
        self.assertEqual([r['history_size_after'] for r in items], [0, 1, 1, 2, 2, 3])
        self.assertEqual([r['warm_up'] for r in items], [True] * 4 + [False] * 2)
        self.assertTrue(all(r['predicted_effort'] is None for r in items[:4]))
        self.assertIsNotNone(items[4]['predicted_effort'])
        self.assertIsNone(items[4]['absolute_error'])
        self.assertIsNone(items[4]['cumulative_mae'])
        self.assertEqual(report['final_summary']['scored_arrivals'], 1)
        self.assertEqual(report['final_summary']['missing_outcomes'], 3)
        self.assertEqual(report['final_summary']['observed_outcomes'], 3)

    def test_independent_running_arithmetic_and_duplicate_history(self):
        report = replay(rows())
        errors, observed = [], []
        for item in report['assessments']:
            actual, predicted = item['observed_effort'], item['predicted_effort']
            if actual is not None:
                observed.append(actual)
            if predicted is not None and actual is not None:
                errors.append(abs(actual - predicted))
                self.assertEqual(item['absolute_error'], errors[-1])
            else:
                self.assertIsNone(item['absolute_error'])
            self.assertEqual(item['scored_arrivals'], len(errors))
            self.assertEqual(item['cumulative_absolute_error'], sum(errors))
            self.assertEqual(item['cumulative_mae'], sum(errors) / len(errors) if errors else None)
            self.assertEqual(item['running_observed_effort_summary']['count'], len(observed))
            self.assertEqual(item['running_observed_effort_summary']['sum'], sum(observed) if observed else None)
        # Identical predictor configurations still become usable prior history.
        self.assertFalse(report['assessments'][2]['warm_up'])
        self.assertEqual(report['final_summary']['cumulative_mae'], sum(errors) / len(errors))
        self.assertEqual(report['final_summary']['observed_effort_summary']['mean'], sum(observed) / len(observed))

    def test_future_changes_cannot_change_earlier_assessments(self):
        data = rows()
        expected = replay(data)
        changed = copy.deepcopy(data)
        for row in changed[3:]:
            row.update(dict.fromkeys(PREDICTORS, 'new_future_category'))
            row['Effort-'] = 999999.
        self.assertEqual(replay(changed)['assessments'][:3], expected['assessments'][:3])
        self.assertEqual(replay(data[:3])['assessments'], expected['assessments'][:3])

    def test_current_target_cannot_change_its_prediction(self):
        data = rows()[:3]
        expected = replay(data)['assessments'][-1]
        for value in [999999., None]:
            changed = copy.deepcopy(data)
            changed[-1]['Effort-'] = value
            result = replay(changed)['assessments'][-1]
            for key in ['predicted_effort', 'history_size_before', 'warm_up', 'predictors']:
                self.assertEqual(result[key], expected[key])
            self.assertNotEqual(result['observed_effort'], expected['observed_effort'])

    def test_reproducibility_and_excluded_fields(self):
        data = rows()
        report = replay(data)
        self.assertEqual(replay(data), report)
        changed = copy.deepcopy(data)
        for row in changed:
            row.update({'Kloc+': 900, 'Defects-': 800, 'YearX': 9999, 'unknown': 'x'})
        self.assertEqual(replay(changed), report)
        json.dumps(report, allow_nan=False)

    def test_lifecycle_validation_and_detached_snapshots(self):
        flow = ProposalFlow()
        with self.assertRaises(ValueError):
            flow.reveal(1)
        for forbidden in ['Effort-', 'Kloc+', 'Defects-', 'idX', 'futureX', 'unknown']:
            with self.subTest(field=forbidden), self.assertRaises(ValueError):
                flow.receive({forbidden: 1})
        pending = flow.receive({'prec': ' n '})
        pending['predictors']['prec'] = 'mutated'
        with self.assertRaises(ValueError):
            flow.receive({})
        with self.assertRaises(ValueError):
            flow.report()
        for invalid in [True, -1, float('nan'), float('inf'), '2']:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                flow.reveal(invalid)
        first = flow.reveal(10)
        self.assertEqual(first['predictors']['prec'], 'n')
        first['predictors']['prec'] = 'mutated'
        report = flow.report()
        self.assertEqual(report['assessments'][0]['predictors']['prec'], 'n')
        report['assessments'].clear()
        self.assertEqual(flow.report()['final_summary']['arrivals'], 1)

    def test_empty_all_missing_and_single_observation(self):
        self.assertEqual(replay([])['final_summary']['arrivals'], 0)
        report = replay([{'Effort-': None}] * 3)
        self.assertEqual(report['final_summary']['missing_outcomes'], 3)
        self.assertIsNone(report['final_summary']['cumulative_mae'])
        self.assertEqual(report['final_summary']['observed_effort_summary']['count'], 0)
        report = replay([{'Effort-': 7}])
        self.assertEqual(report['final_summary']['observed_outcomes'], 1)
        self.assertIsNone(report['assessments'][0]['predicted_effort'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
