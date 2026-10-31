import math
import random
import sys
import unittest
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'source'))
import selector

def row(i, mre=.2, acc=.5, pred=.4, n=40):
    return {'MRE-': str(mre), 'ACC+': str(acc), 'PRED40+': str(pred), 'N_estimators': str(n),
            'criterion': 'squared_error', 'Min_sample_leaves': '2', 'Min_impurity_decrease': '0.5', 'Max_depth': '4', 'id': str(i)}

class SelectorTests(unittest.TestCase):
    def test_order_and_tiebreakers(self):
        rows = [row(1,.2,.5,.4), row(2,.1,.4,.4), row(3,.1,.6,.3), row(4,.1,.6,.5), row(5,.1,.6,.5)]
        self.assertEqual(selector.select(rows)['id'], '4')
    def test_eligibility_cap_and_identifiers(self):
        rows = [row(10,n=100), row(11,n=101), row(12,n=40)]
        self.assertEqual([x['id'] for x in selector.eligible_rows(rows)], ['10','12'])
        self.assertEqual(selector.eligible_rows(rows)[0]['original_record'], 2)
    def test_rng_reproducible_and_independent(self):
        rows = [row(i, mre=i) for i in range(8)]
        a = selector.baseline(rows, seed=42); b = selector.baseline(rows, seed=42)
        self.assertEqual(a['original_record'], b['original_record'])
        self.assertEqual(selector.baseline(rows, seed=42)['original_record'], random.Random(42).choice(selector.eligible_rows(rows))['original_record'])
    def test_summary_arithmetic(self):
        rows = [row(1, .2), row(2, .5), row(3, .2)]
        out = selector.compare(rows, seeds=[1,2,3])
        ds = [r['difference'] for r in out['records']]
        self.assertAlmostEqual(out['mean_difference'], sum(ds)/3)
        self.assertAlmostEqual(out['median_difference'], sorted(ds)[1])
        self.assertEqual(out['wins'] + out['ties'] + out['losses'], 3)
    def test_invalid_and_empty(self):
        with self.assertRaises(ValueError): selector.eligible_rows([], math.inf)
        bad = row(1); bad['criterion'] = ''; self.assertEqual(selector.eligible_rows([bad]), [])
        bad = row(2); bad['MRE-'] = 'nan'; self.assertEqual(selector.eligible_rows([bad]), [])
        with self.assertRaises(ValueError): selector.select([])

if __name__ == '__main__': unittest.main()
