import pathlib, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / 'source'))
import nasa93_summary as ns

class FlowTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile('w', delete=False, newline='')
        f.write('idX,' + ','.join(ns.CATEGORICAL) + ',Effort-\n')
        for i, (value, effort) in enumerate(rows, 1):
            f.write(str(i) + ',' + ','.join(value) + ',' + str(effort) + '\n')
        f.close(); return f.name

    def test_prefix_order_warmup_means_and_same_evaluable_arrivals(self):
        p = self.write([(['a']*22, 0), (['b']*22, 10), (['c']*22, 20)])
        r = ns.flow(p, neighbors=1)
        self.assertEqual(r['warmup_count'], 1); self.assertEqual(r['evaluated_count'], 2)
        self.assertEqual([x['baseline_prediction'] for x in r['records']], [None, 0.0, 5.0])
        self.assertAlmostEqual(r['baseline_mae'], 12.5)

    def test_current_and_future_outcomes_do_not_change_earlier_prediction(self):
        p1 = self.write([(['a']*22, 0), (['b']*22, 10), (['c']*22, 20)])
        p2 = self.write([(['a']*22, 0), (['b']*22, 999), (['c']*22, -500)])
        a, b = ns.flow(p1), ns.flow(p2)
        self.assertEqual(a['records'][1]['prediction'], b['records'][1]['prediction'])
        a['records'][0]['history_count_before'] = 99
        self.assertEqual(b['records'][0]['history_count_before'], 0)

    def test_invalid_missing_nonfinite_zero_and_positions(self):
        p = self.write([(['a']*22, 0), (['b']*22, 'NaN'), (['c']*22, '?'), (['d']*22, 2)])
        r = ns.flow(p)
        self.assertEqual(r['invalid_outcome_count'], 2); self.assertEqual(r['records'][1]['position'], 3)
        self.assertEqual(r['records'][0]['actual_revealed'], 0.0)
        self.assertEqual(len({x['position'] for x in r['records']}), 4)

if __name__ == '__main__': unittest.main()
