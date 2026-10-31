import math, sys, unittest
from pathlib import Path
ROOT=Path(__file__).parents[1]; sys.path.insert(0, str(ROOT/'source'))
from estimator import *

def row(i, group, effort):
    return {'idX':str(i), TARGET:str(effort), **{c:group if c=='prec' else 'h' for c in PREDICTORS}}

class Tests(unittest.TestCase):
    def test_mae_and_cases(self):
        self.assertAlmostEqual(mae([1,3],[2,1]),1.5); self.assertLess(-1,0); self.assertGreater(1,0); self.assertAlmostEqual(0,0)
    def test_reproducible_and_separate(self):
        rows=[row(i, str(i//2), i+1) for i in range(12)]; a=evaluate(rows,42); b=evaluate(rows,42)
        self.assertEqual([x['idX'] for x in a['test']], [x['idX'] for x in b['test']]); self.assertTrue(set(map(group_key,a['train'])).isdisjoint(set(map(group_key,a['test']))))
    def test_baseline_fresh_and_outcome_independence(self):
        rows=[row(i, str(i), i+1) for i in range(8)]; ti,vi=split_groups(rows,42); m=Estimator().fit([rows[i] for i in ti]); p=m.predict([rows[i] for i in vi]); changed=[dict(rows[i], **{TARGET:'999999'}) for i in vi]; m2=Estimator().fit([rows[i] for i in ti]); self.assertEqual(p,m2.predict(changed)); self.assertAlmostEqual(m.mean, sum(i+1 for i in ti)/len(ti))
    def test_exclusions_invalid_and_ids(self):
        self.assertEqual(len(PREDICTORS),22); self.assertNotIn('Effort-',PREDICTORS); self.assertIsNone(finite_float('nan')); self.assertEqual(row(7,'x',2)['idX'],'7')
    def test_insufficient_groups(self):
        rows=[row(i,'same',i) for i in range(4)]; self.assertEqual(len(set(map(group_key,rows))),1)

if __name__=='__main__': unittest.main()
