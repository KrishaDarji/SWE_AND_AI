"""Step 10 comparisons, counted objective boundary, and all-runs audit separation."""

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
import choose
import race
from remember import OBJECTIVE


class RaceTests(unittest.TestCase):
    def pool(self, n=12):
        return tuple(choose.Candidate(i, (float(i),)) for i in range(1, n + 1))

    def factory(self, instances, constant=False):
        class Oracle:
            def __init__(self, budget):
                self.budget, self.calls = budget, []
                instances.append(self)

            @property
            def count(self):
                return len(self.calls)

            def __call__(self, row):
                if self.count >= self.budget or row in self.calls:
                    raise AssertionError('Budget or uniqueness violation')
                self.calls.append(row)
                return 3.0 if constant else float((row * 7) % 13)
        return Oracle

    def test_equal_budgets_fresh_state_histories_and_best(self):
        for budget in (1, 5, 8, 12):
            instances = []
            runs = race.search_runs(('x',), self.pool(), budget, (42, 43), self.factory(instances))
            self.assertEqual(len(instances), 4)
            self.assertEqual(len({id(o) for o in instances}), 4)
            for run, oracle in zip(runs, instances):
                self.assertEqual(run['evaluation_count'], budget)
                self.assertEqual(oracle.calls, run['evaluation_order'])
                self.assertEqual(len(set(oracle.calls)), budget)
                self.assertGreaterEqual(run['search_runtime_seconds'], 0)
                history = run['evaluation_history']
                best = min(history, key=lambda h: h[OBJECTIVE])
                self.assertEqual(run['selected_candidate']['data_row'], best['data_row'])
                for i, event in enumerate(history, 1):
                    prefix = min(history[:i], key=lambda h: h[OBJECTIVE])
                    self.assertEqual(event['best_so_far']['data_row'], prefix['data_row'])

    def test_reproducibility_excludes_wall_time(self):
        def run(seeds):
            runs = race.search_runs(('x',), self.pool(), 8, seeds, self.factory([]))
            for item in runs:
                item.pop('search_runtime_seconds')
            return runs
        first = run((42, 43))
        self.assertEqual(first, run((42, 43)))
        # Reversing repeats does not carry any model, RNG or incumbent state.
        reverse = run((43, 42))
        self.assertEqual(first[:2], reverse[2:])
        self.assertNotEqual(first[1]['evaluation_order'], first[3]['evaluation_order'])

    def test_baseline_matches_uniform_sample_full_pool_and_input_unchanged(self):
        pool = self.pool()
        before = copy.deepcopy(pool)
        for budget in (1, 7, 12):
            oracle = self.factory([])(budget)
            result = race.uniform_search(('x',), pool[::-1], oracle, budget=budget, seed=42)
            expected = [c.data_row for c in random.Random(42).sample(pool, budget)]
            self.assertEqual(oracle.calls, expected)
            self.assertEqual(result['candidate_count'], len(pool))
        self.assertEqual(pool, before)

    def test_ties_first_observed_and_exact_pairing(self):
        runs = race.search_runs(('x',), self.pool(), 8, (42, 43), self.factory([], constant=True))
        for run in runs:
            self.assertEqual(run['selected_candidate']['data_row'], run['evaluation_order'][0])
        audits = [{'absolute_gap': 0.0} for _ in runs]
        paired = race.summarize(runs, audits)['paired_observed']
        self.assertEqual((paired['wins'], paired['ties'], paired['losses']), (0, 2, 0))
        # Near equality is not a tie; test both directions without statistical inference.
        runs[0]['selected_candidate'][OBJECTIVE] = 2.999999999999
        paired = race.summarize(runs, audits)['paired_observed']
        self.assertEqual((paired['wins'], paired['ties'], paired['losses']), (1, 1, 0))
        runs[0]['selected_candidate'][OBJECTIVE] = 3.000000000001
        paired = race.summarize(runs, audits)['paired_observed']
        self.assertEqual((paired['wins'], paired['ties'], paired['losses']), (0, 1, 1))

    def test_invalid_experiment_inputs_before_oracle_construction(self):
        def forbidden(_):
            self.fail('Invalid input created an oracle')
        for budget in (0, -1, True, 1.5, '2', None, 13):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                race.search_runs(('x',), self.pool(), budget, (42,), forbidden)
        for seeds in ((), (42, 42), (-1,), (True,), (2**32,), (1.2,), (42, 'bad')):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                race.search_runs(('x',), self.pool(), 1, seeds, forbidden)
        for pool in ((), (choose.Candidate(1, (1., 2.)),), self.pool()[:1] * 2,
                     (choose.Candidate(0, (1.,)),), (choose.Candidate(1, (float('nan'),)),),
                     ({'data_row': 1, OBJECTIVE: 3},)):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                race.search_runs(('x',), pool, 1, (42,), forbidden)
        with self.assertRaises(ValueError):
            race.uniform_search((OBJECTIVE,), self.pool(), lambda _: self.fail(), budget=1, seed=42)

    def test_both_methods_only_call_objective_capability(self):
        for search in (choose.select, race.uniform_search):
            calls = []

            class CallOnly:
                def __getattribute__(self, name):
                    raise AssertionError(f'Inspected objective attribute: {name}')

                def __call__(self, row):
                    calls.append(row)
                    return float(row % 4)

            with patch('remember.load_measurements', side_effect=AssertionError('Bulk access')):
                result = search(('x',), self.pool(), CallOnly(), budget=8, seed=42)
            self.assertEqual(result['actual_evaluation_count'], len(calls))

    def test_unqueried_targets_do_not_affect_either_search(self):
        for search in (choose.select, race.uniform_search):
            with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
                path = Path(directory) / 'data.csv'

                def write(hidden):
                    path.write_text('x,Performance-\n' + ''.join(
                        f'{i},{hidden.get(i, float(i % 4))}\n' for i in range(1, 13)))

                write({})
                settings, pool = choose.load_candidates(path)
                first = search(settings, pool, choose.CSVOracle(path, 8), budget=8, seed=42)
                queried = {h['data_row'] for h in first['evaluation_history']}
                write({i: 'HIDDEN_NONNUMERIC' for i in range(1, 13) if i not in queried})
                self.assertEqual((settings, pool), choose.load_candidates(path))
                second = search(settings, pool, choose.CSVOracle(path, 8), budget=8, seed=42)
                self.assertEqual(first, second)

    def test_reused_oracle_and_misreported_count_are_rejected(self):
        oracle = self.factory([])(8)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            race.search_runs(('x',), self.pool(), 8, (42,), lambda _: oracle)
        original = race.uniform_search

        def bad(*args, **kwargs):
            result = original(*args, **kwargs)
            result['actual_evaluation_count'] -= 1
            return result

        with patch.object(race, 'uniform_search', bad), self.assertRaisesRegex(RuntimeError, 'budget'):
            race.search_runs(('x',), self.pool(), 8, (42,), self.factory([]))

    def test_all_runs_frozen_before_first_audit_and_audit_cannot_change_search(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            root = Path(directory)
            dataset = root / 'data.csv'
            dataset.write_text('x,Performance-\n0,10\n1,4\n2,1\n3,8\n')
            before = dataset.read_bytes()
            original = race.post_selection_audit
            observed_frozen, calls = [], []

            def audit(path, selection):
                frozen = json.loads((root / 'out/frozen_selections.json').read_text())
                self.assertEqual(len(frozen), 4)
                self.assertTrue(all(r['evaluation_count'] == 2 for r in frozen))
                self.assertNotIn('audit_results', frozen[0])
                self.assertIn(json.loads(selection), frozen)
                observed_frozen.append(frozen)
                calls.append(json.loads(selection))
                return original(path, selection)

            with patch.object(race, 'post_selection_audit', side_effect=audit):
                result = race.run_race(dataset, root / 'out', budget=2, seeds=(42, 43))
            self.assertEqual(result['runs'], observed_frozen[0])
            self.assertEqual(len(calls), 4)
            for run, audit_result in zip(result['runs'], result['audit_results']):
                frozen_run = json.dumps(run, sort_keys=True, allow_nan=False)
                self.assertEqual(audit_result['selection_sha256'], hashlib.sha256(frozen_run.encode()).hexdigest())
                self.assertEqual(audit_result['dataset_minimum'], 1)
                self.assertEqual(audit_result['absolute_gap'], run['selected_candidate'][OBJECTIVE] - 1)
                self.assertFalse(audit_result['fed_back_into_selection'])
            self.assertEqual(dataset.read_bytes(), before)
            self.assertEqual(result, json.loads((root / 'out/race.json').read_text()))
            self.assertTrue((root / 'out/per_seed.csv').is_file())
            self.assertEqual(result['statistical_analysis']['status'], 'pending')
            self.assertFalse(result['statistical_analysis']['substitute_test_performed'])

    def test_audit_output_cannot_feed_back_into_selections(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            root = Path(directory)
            dataset = root / 'data.csv'
            dataset.write_text('x,Performance-\n0,10\n1,4\n2,1\n')
            first = race.run_race(dataset, root / 'one', budget=2, seeds=(42,))
            with patch.object(race, 'post_selection_audit', return_value={'absolute_gap': 999999}):
                second = race.run_race(dataset, root / 'two', budget=2, seeds=(42,))
            for a, b in zip(first['runs'], second['runs']):
                a.pop('search_runtime_seconds')
                b.pop('search_runtime_seconds')
                self.assertEqual(a, b)

    def test_cli_invalid_inputs_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as directory:
            root = Path(directory)
            dataset = root / 'race.json'
            dataset.write_text('x,Performance-\n0,2\n')
            before = dataset.read_bytes()
            cases = [('--budget', '0'), ('--budget', '2'), ('--seeds', '-1'),
                     ('--seeds', '42', '42'), ('--seeds',)]
            for options in cases:
                with self.subTest(options=options), self.assertRaises(SystemExit) as caught:
                    race.main([str(dataset), '--output-dir', str(root / 'out'), *options])
                self.assertEqual(caught.exception.code, 2)
            self.assertFalse((root / 'out').exists())
            with self.assertRaises(SystemExit):
                race.main([str(dataset), '--output-dir', directory, '--budget', '1'])
            with self.assertRaises(SystemExit):
                race.main([str(root / 'missing.csv'), '--output-dir', str(root / 'out')])
            self.assertEqual(dataset.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
