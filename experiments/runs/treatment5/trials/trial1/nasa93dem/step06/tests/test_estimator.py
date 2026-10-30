import os, tempfile, unittest
from source.estimator import *

class Tests(unittest.TestCase):
    def rows(self):
        return [{'line':i+2,'id':str(i+1),'x':tuple(('a' if i<2 else 'b') if j==0 else str(i%2) for j in range(22)),'y':float(i+1)} for i in range(4)]
    def proposal(self, first='a', second='0'):
        return {k:(first if i==0 else second if i==1 else 'a') for i,k in enumerate(DESCRIPTORS)}
    def test_prediction_and_training_reference(self):
        m=Estimator(self.rows()[:3]); p=self.proposal()
        z=m.predict(p); self.assertEqual(z['estimate'],1.0); self.assertEqual(z['reference'],2.0)
    def test_excluded_inputs_and_missing_unseen(self):
        m=Estimator(self.rows()[:3]); self.assertNotIn('Kloc+',DESCRIPTORS)
        z=m.predict(self.proposal(' unseen ','9')); self.assertEqual(z['unseen_descriptors'][:2],['prec','flex']); self.assertEqual(len(z['unseen_descriptors']),22)
        with self.assertRaises(ValueError): m.predict({'prec':'a'})
    def test_evidence_and_selection(self):
        m=Estimator(self.rows()[:3]); z=m.predict(self.proposal()); self.assertTrue(z['evidence'].startswith('model_behavior:')); self.assertEqual(min(self.rows()[3:],key=lambda r:(r['line'],r['id']) )['id'],'4')
    def test_outcome_does_not_change_frozen_prediction(self):
        a=self.rows(); b=self.rows(); b[3]['y']=999999
        m1=Estimator(a[:3]); m2=Estimator(b[:3]); self.assertEqual(m1.predict(self.proposal('b','1')),m2.predict(self.proposal('b','1')))
    def test_insufficient_and_ids(self):
        with self.assertRaises(ValueError): Estimator(self.rows()[:1])
        self.assertEqual(load_rows(self._csv())[0]['id'],'7'); self.assertEqual(load_rows(self._csv())[0]['line'],3)
    def _csv(self):
        p=os.path.join(tempfile.gettempdir(),'s6.csv'); keys=','.join(['idX']+DESCRIPTORS+['Effort-']); vals=','.join(['5']+['a']*22+['notnum']); vals2=','.join(['7']+[' a ']*22+['3']); open(p,'w').write(keys+'\n'+vals+'\n'+vals2+'\n'); return p

if __name__=='__main__': unittest.main()
