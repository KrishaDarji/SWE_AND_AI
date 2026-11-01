import csv, os, tempfile, unittest
from fix import fix

H='MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth'
def data(rows):
    f=tempfile.NamedTemporaryFile(mode='w',delete=False,newline=''); f.write(H+'\n');
    for r in rows: f.write(','.join(map(str,r))+'\n')
    f.close(); return f.name
class FixTests(unittest.TestCase):
    def test_min_changes_first(self):
        p=data([[10,.5,.5,1,a,1,1,1] for a in ['x','x']]+[[.4,.4,.4,2,'x',1,1,1],[.3,.9,.9,2,'y',2,2,2]]); r=fix(p); self.assertEqual(r['candidate']['change_count'],1); os.unlink(p)
    def test_ties_and_reports(self):
        p=data([[.8,.5,.5,1,'x',1,1,1],[.7,.4,.6,1,'x',1,1,1],[.7,.5,.7,1,'x',1,1,2]]); r=fix(p); self.assertEqual(r['candidate']['original_position'],3); self.assertIn('changed_fields',r['candidate']); os.unlink(p)
    def test_invalid_first_and_no_improvement(self):
        p=data([["bad",.5,.5,1,'x',1,1,1],[.1,.5,.5,1,'x',1,1,1]]); self.assertEqual(fix(p)['status'],'invalid_baseline'); os.unlink(p)
        p=data([[.1,.5,.5,1,'x',1,1,1],[.1,.6,.5,2,'x',1,1,1]]); self.assertEqual(fix(p)['status'],'no_improving_candidate'); os.unlink(p)
if __name__=='__main__': unittest.main()
