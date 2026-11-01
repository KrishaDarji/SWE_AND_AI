import tempfile, unittest
from source.analyzer import justify

H='MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n'
def fixture(s):
    f=tempfile.NamedTemporaryFile('w',delete=False,newline=''); f.write(s); f.close(); return f.name
def row(m,a,p,n='1',c='x',l='1',i='0.1',d='3'): return f'{m},{a},{p},{n},{c},{l},{i},{d}\n'

class JustifyTests(unittest.TestCase):
    def test_all_tiebreakers_and_full_retention(self):
        s=H+row(1,1,1,'1')+row(1,2,1,'2')+row(1,2,3,'3')+row(1,2,3,'4')
        x=justify(fixture(s)); self.assertEqual(x['recommendation']['original_position'],4); self.assertEqual(x['recommendation']['configuration']['N_estimators'],3)
    def test_finite_eligibility_missing_and_positions(self):
        x=justify(fixture(H+row('nan',9,9,'bad')+row(2,4,6,'2')+'2,?,6,3,x,1,0.1,3\n'))
        self.assertEqual(x['recommendation']['original_position'],3); self.assertEqual(x['malformed_row_positions'],[])
    def test_independent_even_medians_and_directions(self):
        s=H+row(1,10,100)+row(3,20,200)+row(2,99,300)
        x=justify(fixture(s)); self.assertEqual(x['references']['MRE-']['finite_count'],3); self.assertEqual(x['references']['MRE-']['median'],2); self.assertEqual(x['references']['ACC+']['median'],20); self.assertEqual(x['comparisons']['MRE-']['direction'],'minimize')
    def test_empty_and_no_eligible(self):
        self.assertEqual(justify(fixture(''))['status'],'empty_data'); self.assertEqual(justify(fixture(H+row('?',1,1)))['status'],'no_eligible_recommendation')
    def test_association_is_global_context(self):
        x=justify(fixture(H+row(1,2,3)+row(2,3,4,'2','y'))); self.assertIn('global marginal',x['association_context']['label']); self.assertIn('numeric_groups',x['association_context']['grouping_conventions'])
if __name__=='__main__': unittest.main()
