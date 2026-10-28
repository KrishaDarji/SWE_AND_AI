"""Online prefix correctness, isolation, validation, and batch agreement."""

import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / 'source'))
import flow
from remember import load_measurements, summarize


class FlowTests(unittest.TestCase):
    def test_independent_prefix_statistics(self):
        tracker = flow.Flow(('flag',))
        values = [12, 3, 8, -4, 3, 20]
        for i, value in enumerate(values, 1):
            state = tracker.update({'flag': i}, value)
            prefix = list(map(Fraction, values[:i]))
            mean = sum(prefix) / i
            variance = sum((x - mean) ** 2 for x in prefix) / i
            self.assertEqual(state['count'], i)
            self.assertAlmostEqual(state['mean'], float(mean))
            self.assertAlmostEqual(state['population_variance'], float(variance))
            self.assertAlmostEqual(state['population_stddev'], math.sqrt(variance))
            best_index = values[:i].index(min(values[:i])) + 1
            self.assertEqual(state['best']['arrival_index'], best_index)
            self.assertEqual(state['best']['settings'], {'flag': best_index})
            self.assertEqual(state['best']['Performance-'], min(values[:i]))

    def test_empty_single_ties_and_new_best(self):
        tracker = flow.Flow(('x',))
        self.assertEqual(tracker.snapshot(), {
            'count': 0, 'mean': None, 'population_variance': None,
            'population_stddev': None, 'best': None, 'best_tie_count': 0})
        first = tracker.update({'x': 1}, 7)
        self.assertEqual((first['mean'], first['population_variance'],
                          first['population_stddev']), (7, 0, 0))
        for x, y in ((2, 9), (3, 7), (1, 7)):
            state = tracker.update({'x': x}, y)
        self.assertEqual(state['best'], first['best'])
        self.assertEqual(state['best_tie_count'], 3)
        state = tracker.update({'x': 4}, 2)
        self.assertEqual(state['best']['arrival_index'], 5)
        self.assertEqual(state['best_tie_count'], 1)

    def test_invalid_schema_values_and_atomic_rejection(self):
        for names in ([], ['x', 'x'], [''], [' '], ['Performance-'], [1], 'x', None):
            with self.subTest(names=names), self.assertRaises(ValueError):
                flow.Flow(names)
        tracker = flow.Flow(('x',))
        tracker.update({'x': 1}, 2)
        before = tracker.snapshot()
        bad = [None, True, '3', 2j, float('inf'), float('-inf'), float('nan'), 10 ** 400]
        cases = [({}, 3), ({'x': 1, 'Performance-': 3}, 3), ([], 3)]
        cases += [({'x': value}, 3) for value in bad]
        cases += [({'x': 1}, value) for value in bad]
        for config, result in cases:
            with self.subTest(config=config, result=result), self.assertRaises(ValueError):
                tracker.update(config, result)
            self.assertEqual(tracker.snapshot(), before)
        large = flow.Flow(('x',))
        large.update({'x': 0}, 1e308)
        before = large.snapshot()
        with self.assertRaisesRegex(ValueError, 'range'):
            large.update({'x': 0}, -1e308)
        self.assertEqual(large.snapshot(), before)

    def test_snapshots_and_input_are_detached(self):
        tracker = flow.Flow(('x',))
        config = {'x': 2}
        first = tracker.update(config, 3)
        config['x'] = 999
        first['best']['settings']['x'] = 998
        self.assertEqual(tracker.snapshot()['best']['settings'], {'x': 2})
        tracker.update({'x': 3}, 1)
        self.assertEqual(first['count'], 1)
        self.assertEqual(first['mean'], 3)

    def test_future_changes_cannot_change_earlier_outputs(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            a, b = Path(tmp) / 'a.csv', Path(tmp) / 'b.csv'
            prefix = 'x,Performance-\n0,9\n1,3\n0,5\n'
            a.write_text(prefix + '1,2\n0,1\n')
            b.write_text(prefix + '90,2000\n91,-1000\n92,7\n')
            first, changed = list(flow.replay_csv(a)), list(flow.replay_csv(b))
            self.assertEqual(first[:3], changed[:3])
            b.write_text(prefix + 'future,invalid\n')
            arrivals = flow.replay_csv(b)
            self.assertEqual([next(arrivals) for _ in range(3)], first[:3])
            with self.assertRaises(ValueError):
                next(arrivals)

    def test_lazy_csv_validation_and_objective_column_order(self):
        cases = ['', 'x,x,Performance-\n1,1,2\n', 'x\n1\n',
                 'Performance-\n3\n', ',Performance-\n1,2\n',
                 'x,Performance-\n', 'x,Performance-\n1\n',
                 'x,Performance-\n1,2,3\n', 'x,Performance-\n1,nan\n',
                 'x,Performance-\ninf,2\n', 'x,Performance-\n,2\n']
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            path = Path(tmp) / 'data.csv'
            for text in cases:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    list(flow.replay_csv(path))
            path.write_text('\ufeffPerformance-,x\n 7 ,1\n7,1\n', encoding='utf-8')
            updates = list(flow.replay_csv(path))
            self.assertEqual(updates[-1]['summary']['count'], 2)
            self.assertEqual(updates[-1]['summary']['best_tie_count'], 2)
            self.assertEqual(updates[0]['observed']['settings'], {'x': 1})

    def test_actual_stream_prefixes_final_batch_and_cli(self):
        dataset = Path(os.environ['X264_DATASET'])
        before = dataset.read_bytes()
        with dataset.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        values = [float(row['Performance-']) for row in rows]
        result = flow.report(dataset)
        self.assertIn('not evidence of real chronological order', result['arrival_order_rule'])
        for index, update in enumerate(result['history'], 1):
            prefix = values[:index]
            state = update['summary']
            self.assertEqual(state['count'], index)
            self.assertAlmostEqual(state['mean'], statistics.mean(prefix), places=9)
            self.assertAlmostEqual(state['population_stddev'], statistics.pstdev(prefix), places=9)
            best_index = prefix.index(min(prefix))
            expected = {key: float(value) for key, value in rows[best_index].items()
                        if key != 'Performance-'}
            self.assertEqual(state['best']['settings'], expected)
            self.assertEqual(state['best']['arrival_index'], best_index + 1)
            self.assertEqual(state['best']['Performance-'], min(prefix))
        batch = summarize(load_measurements(dataset))
        final = result['final_summary']
        self.assertEqual(final['count'], batch['row_count'])
        self.assertAlmostEqual(final['mean'], batch['objective']['statistics']['mean'])
        self.assertAlmostEqual(final['population_stddev'],
                               batch['objective']['statistics']['population_stddev'])
        self.assertEqual(final['best']['Performance-'], batch['objective']['statistics']['min'])
        self.assertEqual(final['best_tie_count'], len(batch['best_runs']))
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            process = subprocess.run([sys.executable, '-B', str(STEP / 'source/flow.py'),
                                      str(dataset), '--output-dir', tmp], text=True,
                                     capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads((Path(tmp) / 'flow.json').read_text()), result)
        self.assertEqual(dataset.read_bytes(), before)

    def test_cli_rejects_overwrite_and_bad_input(self):
        with tempfile.TemporaryDirectory(dir=STEP / 'tests') as tmp:
            dataset = Path(tmp) / 'flow.json'
            dataset.write_text('x,Performance-\n1,3\n')
            before = dataset.read_bytes()
            with self.assertRaises(SystemExit) as error:
                flow.main([str(dataset), '--output-dir', tmp])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(dataset.read_bytes(), before)
            dataset = Path(tmp) / 'bad.csv'
            dataset.write_text('x,Performance-\n1,nan\n')
            with self.assertRaises(SystemExit) as error:
                flow.main([str(dataset), '--output-dir', str(Path(tmp) / 'output')])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(tmp) / 'output/flow.json').exists())


if __name__ == '__main__':
    unittest.main()
