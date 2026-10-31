import copy, math, os, tempfile, unittest
from source.justify_estimator import *

class JustifyTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'original_id':str(i),'row_number':i+2,'x':tuple(['a' if i%2 else 'b']*22),'y':float(i+1)} for i in range(8)]
        self.model=fit(self.rows)
        self.p=dict(zip(DESCRIPTORS,self.rows[0]['x']))
    def test_consistency_and_exclusions(self):
        a,d=predict(self.model,self.p); e=explain(self.model,self.p,a,d)
        self.assertEqual(a,e['estimate']); self.assertNotIn('Kloc+',self.p); self.assertNotIn('Effort-',self.p); self.assertEqual(len(self.p),22)
    def test_training_reference_and_labels(self):
        self.assertEqual(median_reference(self.rows),4.5); self.assertIn('training-only',explain(self.model,self.p,*predict(self.model,self.p))['reference_label']); self.assertTrue(all('label' in x for x in associations(self.rows)))
    def test_missing_unseen_and_selection(self):
        q=dict(self.p); del q[DESCRIPTORS[0]]
        with self.assertRaises(ValueError): predict(self.model,q)
        q[DESCRIPTORS[0]]='unseen'; value,_=predict(self.model,q); self.assertTrue(math.isfinite(value))
        self.assertEqual(min(self.rows,key=lambda r:r['row_number'])['original_id'],'0')
    def test_outcome_does_not_change_frozen_prediction(self):
        a,d=predict(self.model,self.p); frozen=explain(self.model,self.p,a,d); changed=copy.deepcopy(self.model); changed['rows'][0]['y']=999999
        b,_=predict(changed,self.p); self.assertNotEqual(a,b); self.assertEqual(frozen['estimate'],a)
    def test_invalid_targets_and_insufficient(self):
        fd,path=tempfile.mkstemp(text=True); os.close(fd)
        try:
            with open(path,'w') as f: f.write('idX,'+','.join(DESCRIPTORS)+',Effort-\n1,'+','.join(['a']*22)+',bad\n2,'+','.join(['b']*22)+',4\n')
            got=read_rows(path); self.assertEqual([x['original_id'] for x in got],['2'])
        finally: os.unlink(path)
        with self.assertRaises(ValueError): fit([])

if __name__ == '__main__': unittest.main()
