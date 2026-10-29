"""Independent prefix/batch checks and streaming interface contracts."""
import copy
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from flow import FlowEvaluator, iter_arrivals, replay
from summarize import CONFIGURATION, OBJECTIVES

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')
HEADERS = list(OBJECTIVES) + list(CONFIGURATION)


class FlowTests(unittest.TestCase):
    def assert_batch(self, summary, records):
        self.assertEqual(summary['configurations_received'], len(records))
        for name, direction in OBJECTIVES.items():
            state = summary['objectives'][name]
            observed = [(i + 1, c, o[name]) for i, (c, o) in enumerate(records)
                        if o.get(name) is not None]
            values = [v for _, _, v in observed]
            self.assertEqual(state['observed_count'], len(values))
            self.assertEqual(state['missing_count'], len(records) - len(values))
            if not values:
                self.assertIsNone(state['mean'])
                self.assertIsNone(state['best'])
                self.assertEqual(state['best_tie_count'], 0)
            else:
                self.assertTrue(math.isclose(state['mean'], statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12))
                best = (min if direction == 'minimize' else max)(values)
                first = next((i, c, v) for i, c, v in observed if v == best)
                self.assertEqual(state['best']['value'], best)
                self.assertEqual(state['best']['arrival_index'], first[0])
                self.assertEqual(state['best']['configuration'], first[1])
                self.assertEqual(state['best_tie_count'], values.count(best))
            for key, fn in [('variance_sample', statistics.variance), ('std_sample', statistics.stdev)]:
                if len(values) < 2:
                    self.assertIsNone(state[key])
                else:
                    self.assertTrue(math.isclose(state[key], fn(values), rel_tol=1e-10, abs_tol=1e-12), (name, key))

    def test_independent_statistics_after_every_arrival(self):
        f = FlowEvaluator()
        records = []
        for i in range(1, 61):
            config = dict.fromkeys(CONFIGURATION)
            config['N_estimators'] = float(i)
            outcomes = {'MRE-': None if i % 4 == 0 else i / 13,
                        'ACC+': None if i % 3 == 0 else (-1) ** i * i,
                        'PRED40+': None if i % 5 == 0 else 0.7}
            records.append((config, outcomes))
            self.assert_batch(f.update(config, outcomes), records)

    def test_empty_missing_single_and_zero_variance(self):
        f = FlowEvaluator()
        self.assert_batch(f.snapshot(), [])
        a = f.update({}, {})
        for s in a['objectives'].values():
            self.assertEqual(s['missing_count'], 1)
            self.assertIsNone(s['mean'])
        b = f.update({}, {'MRE-': ' 2 ', 'ACC+': ' ', 'PRED40+': None})
        self.assertEqual(b['objectives']['MRE-']['mean'], 2)
        self.assertIsNone(b['objectives']['MRE-']['variance_sample'])
        c = f.update({}, {'MRE-': 2})
        self.assertEqual(c['objectives']['MRE-']['variance_sample'], 0)
        self.assertEqual(c['objectives']['ACC+']['missing_count'], 3)

    def test_directions_ties_and_separate_best_configurations(self):
        f = FlowEvaluator()
        for i, values in enumerate([(3, 8, 1), (1, 5, 9), (1, 10, 4), (2, 10, 9)]):
            s = f.update({'N_estimators': i}, dict(zip(OBJECTIVES, values)))
        states = s['objectives']
        self.assertEqual([states[n]['best']['arrival_index'] for n in OBJECTIVES], [2, 3, 2])
        self.assertEqual([states[n]['best_tie_count'] for n in OBJECTIVES], [2, 2, 2])
        self.assertEqual(states['ACC+']['best']['configuration']['N_estimators'], 2)
        f.update({}, {'MRE-': -1})
        self.assertEqual(f.snapshot()['objectives']['MRE-']['best_tie_count'], 1)

    def test_invalid_arrivals_are_atomic(self):
        f = FlowEvaluator()
        f.update({}, {'MRE-': 1})
        before = f.snapshot()
        bad = [(None, {}), ([], {}), ({'MRE-': 2}, {}), ({'criterion': 3}, {}),
               ({'Max_depth': True}, {}), ({'Max_depth': 'NaN'}, {}),
               ({}, None), ({}, []), ({}, {'unknown': 1}), ({}, {'Max_depth': 1})]
        bad += [({}, {'MRE-': 2, 'ACC+': x}) for x in [True, [], {}, 'bad', math.inf, math.nan, 'NaN', 'inf', 10**1000]]
        for config, outcome in bad:
            with self.subTest(config=config, outcome=outcome):
                with self.assertRaises(ValueError):
                    f.update(config, outcome)
                self.assertEqual(f.snapshot(), before)

    def test_overflow_is_atomic(self):
        f = FlowEvaluator()
        f.update({}, {'MRE-': 1e308})
        before = f.snapshot()
        with self.assertRaises(ValueError):
            f.update({}, {'MRE-': -1e308})
        self.assertEqual(f.snapshot(), before)

    def test_detached_inputs_snapshots_and_no_retained_history(self):
        f = FlowEvaluator()
        config, outcomes = {'criterion': 'a'}, {'ACC+': 5}
        first = f.update(config, outcomes)
        saved = copy.deepcopy(first)
        config['criterion'] = 'mutated'
        outcomes['ACC+'] = 999
        f.update({}, {'ACC+': 6})
        self.assertEqual(first, saved)
        first['objectives']['ACC+']['best']['configuration']['criterion'] = 'changed'
        current = f.snapshot()
        current['objectives']['ACC+']['best']['value'] = -9
        self.assertEqual(f.snapshot()['objectives']['ACC+']['best']['value'], 6)
        self.assertEqual(set(vars(f)), {'_received', '_state'})

    def write_csv(self, path, rows, headers=HEADERS):
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_future_changes_do_not_change_earlier_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            a, b = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            prefix = [[3, 2, 1, 40, ' a ', 1, 0, 5], [2, '', 4, 50, 'b', 1, 0, 6]]
            self.write_csv(a, prefix + [[1, 9, 5, 60, 'a', 1, 0, 7]])
            self.write_csv(b, prefix + [[999, -10, -9, 90, 'z', 2, 1, 8]])
            ra, rb = replay(a), replay(b)
            self.assertEqual(ra['history'][:2], rb['history'][:2])
            self.assertNotEqual(ra['history'][2], rb['history'][2])
            saved = copy.deepcopy(ra['history'][0])
            ra['final_summary']['objectives']['MRE-']['best']['configuration']['criterion'] = 'x'
            ra['history'][1]['summary']['objectives']['MRE-']['mean'] = -99
            self.assertEqual(ra['history'][0], saved)
            # Valid prefix can be consumed even when a later row is invalid.
            self.write_csv(a, prefix + [['bad', 9, 5, 60, 'a', 1, 0, 7]])
            stream = iter_arrivals(a)
            next(stream)
            next(stream)
            with self.assertRaises(ValueError):
                next(stream)

    def test_csv_validation_reordering_and_missing(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            p = Path(directory) / 'a.csv'
            for headers, rows in [(HEADERS[:-1], []), (HEADERS[:-1] + ['MRE-'], []),
                                  (HEADERS, []), (HEADERS, [[1]]),
                                  (HEADERS, [['nan', 1, 2, 3, 'a', 1, 0, 5]])]:
                self.write_csv(p, rows, headers)
                with self.assertRaises(ValueError):
                    replay(p)
            self.write_csv(p, [[''] * 8])
            self.assertEqual(replay(p)['final_summary']['objectives']['ACC+']['missing_count'], 1)
            self.write_csv(p, [list(reversed([1, 2, 3, 40, ' a ', 1, 0, 5]))], HEADERS[::-1])
            self.assertEqual(replay(p)['final_summary']['objectives']['MRE-']['mean'], 1)

    def test_actual_csv_final_statistics_against_independent_batch(self):
        report = replay(DATASET)
        with DATASET.open(newline='', encoding='utf-8-sig') as handle:
            records = []
            for row in csv.DictReader(handle):
                config = {n: (row[n].strip() if n == 'criterion' else float(row[n]))
                          if row[n].strip() else None for n in CONFIGURATION}
                outcomes = {n: float(row[n]) if row[n].strip() else None for n in OBJECTIVES}
                records.append((config, outcomes))
        self.assertEqual(len(report['history']), 10000)
        self.assertEqual(report['history'][-1]['summary'], report['final_summary'])
        self.assert_batch(report['final_summary'], records)
        for count in [1, 2, 37, 100, 999]:
            self.assert_batch(report['history'][count-1]['summary'], records[:count])
        json.dumps(report, allow_nan=False)

    def test_cli_success_and_expected_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as directory:
            p, output = Path(directory) / 'a.csv', Path(directory) / 'flow.json'
            self.write_csv(p, [[1, 2, 3, 40, 'a', 1, 0, 5]])
            cmd = [sys.executable, str(ROOT / 'source/flow.py'), str(p), '--output', str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('not verified chronological order', result.stdout)
            self.assertEqual(json.loads(output.read_text())['final_summary']['configurations_received'], 1)
            p.unlink()
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Error:', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
