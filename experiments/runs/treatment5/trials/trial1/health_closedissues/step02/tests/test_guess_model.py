import math, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'source'))
from guess_model import GuessModel, PREDICTORS, _key

def rows(n=12):
    out=[]
    for i in range(n):
        out.append({'MRE-':str(i/20),'ACC+':str(.5+i/40),'PRED40+':str(i%2),'N_estimators':str(10+i%4),'criterion':['a','b'][i%2],'Min_sample_leaves':str(i%3+1),'Min_impurity_decrease':str(i/10),'Max_depth':str(i%5+1)})
    return out

class T(unittest.TestCase):
    def test_exclusion_grouping_repro_missing_unseen_and_prediction(self):
        m=GuessModel(rows()); self.assertEqual(m.predictor_names,list(PREDICTORS)); self.assertEqual(m.evaluate().keys(),{'MRE-','ACC+'})
        self.assertEqual((m.train_keys|m.test_keys),{tuple([float(x) if j!=1 else x for j,x in enumerate(k)]) for k in m.train_keys|m.test_keys})
        self.assertEqual(GuessModel(rows()).predict({'N_estimators':10,'criterion':'z','Min_sample_leaves':1,'Min_impurity_decrease':0,'Max_depth':1}), GuessModel(rows()).predict({'N_estimators':10,'criterion':'z','Min_sample_leaves':1,'Min_impurity_decrease':0,'Max_depth':1}))
        with self.assertRaises(ValueError): m.predict({'N_estimators':1})
    def test_missing_and_invalid_targets_zero_and_insufficient(self):
        bad=rows(); bad[0]['MRE-']='?'; bad[1]['ACC+']='nan'; bad[2]['MRE-']='0'
        m=GuessModel(bad); self.assertTrue(math.isfinite(m.training_means[0])); self.assertGreaterEqual(m.training_means[0],0)
        one=rows(); [one.__setitem__(i,{**one[i],'N_estimators':'1','criterion':'a','Min_sample_leaves':'1','Min_impurity_decrease':'0','Max_depth':'1'}) for i in range(len(one))]
        with self.assertRaises(ValueError): GuessModel(one)
    def test_holdout_outcomes_do_not_fit(self):
        a=GuessModel(rows()); changed=rows()
        for r in changed:
            if _key(r) in a.test_keys: r['MRE-']='999'; r['ACC+']='999'
        b=GuessModel(changed); self.assertEqual(a.coeffs,b.coeffs); self.assertEqual(a.training_means,b.training_means)
    def test_preprocessing_train_only(self):
        m=GuessModel(rows()); self.assertEqual(m.categories,sorted({k[1] for k in m.train_keys})); self.assertEqual(len(m.categories),2)
if __name__=='__main__': unittest.main()
