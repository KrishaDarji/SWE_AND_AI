import csv, os, tempfile, unittest
from race import race, eligible, classify
FIELDS=['MRE-','ACC+','PRED40+','N_estimators','criterion','Min_sample_leaves','Min_impurity_decrease','Max_depth']
def make(rows):
 f=tempfile.NamedTemporaryFile(mode='w',newline='',delete=False); w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name
def row(m=1,a=1,p=1,n=10): return dict(zip(FIELDS,[m,a,p,n,'gini',1,0,3]))
class RaceTests(unittest.TestCase):
 def tearDown(self):
  for p in getattr(self,'paths',[]): os.unlink(p)
 def data(self,rs):
  p=make(rs); self.paths=getattr(self,'paths',[])+[p]; return p
 def test_same_population_reproducible_and_positions(self):
  p=self.data([row(5),row(1,n=101),row(2),{'MRE-':3,'ACC+':3,'PRED40+':3,'N_estimators':10,'criterion':'gini','Min_sample_leaves':1,'Min_impurity_decrease':0,'Max_depth':3}])
  a=race(p,100,[42,43]); b=race(p,100,[42,43]); self.assertEqual(a,b); self.assertEqual(a['population']['feasible_count'],3); self.assertEqual(a['results'][0]['selector']['position'],4)
 def test_classification_boundaries_and_summary(self):
  self.assertEqual([classify(x,1) for x in (-1.01,-1,0,1,1.01)],['loss','tie','tie','tie','win'])
  p=self.data([row(1),row(3),row(2)]); a=race(p,seeds=[42,43,44]); self.assertEqual(sum(a['summary'][k] for k in ('win','tie','loss')),3); self.assertAlmostEqual(a['summary']['mean_difference'],sum(x['mre_difference'] for x in a['results'])/3)
 def test_empty_single_invalid(self):
  p=self.data([]); self.assertEqual(race(p)['status'],'empty'); self.assertIsNone(race(p)['results'][0]['baseline'])
  p=self.data([row()]); self.assertEqual(race(p)['summary']['tie'],10); self.assertEqual(race(p)['results'][0]['selector']['position'],2)
  with self.assertRaises(ValueError): race(p,cap=-1)
  with self.assertRaises(ValueError): race(p,tolerance=float('inf'))
 def test_secondary_directions(self):
  p=self.data([row(1,1,1),row(2,3,4)]); xs=race(p,seeds=range(42,52))['results']; self.assertTrue(any(x['acc_difference'] == 2 and x['pred40_difference'] == 3 for x in xs))
if __name__=='__main__': unittest.main()
