import math,sys,os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','source')))
from certify import *
def eq(a,b,t=1e-8): assert abs(a-b)<=t*max(1,abs(b)),(a,b)
import unittest
class TestCertify(unittest.TestCase):
 def test_group(self): self.assertTrue(read('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')[0])
 def test_direction(self): self.assertLess(welch([1,2],[0,1])['difference_candidate_minus_reference'],0)
 def test_welch(self):
  r=welch([1,2,3],[2,4,6]); eq(r['difference_candidate_minus_reference'],2); eq(r['degrees_freedom'],50/17); assert 0<r['p_two_sided']<1
 def test_threshold_uncertainty(self):
  r=welch([10,11,9],[7,9,10],1); assert r['point_improvement_beyond_threshold']; assert not r['entire_interval_favorable_beyond_threshold']
 def test_missing(self):
  import tempfile,os
  fd,p=tempfile.mkstemp(text=True); os.write(fd,b'criterion,MRE-,ACC+,PRED40+\nsquared_error,?,1,1\nabsolute_error,nan,1,1\n'); os.close(fd)
  assert read(p)[0]=={}; os.unlink(p)
 def test_insufficient_zero(self): self.assertEqual(welch([1],[2,3])['status'],'undefined'); self.assertEqual(welch([1,1],[2,2])['status'],'undefined')
 def test_numerics(self):
  eq(t_cdf(0,10),.5); eq(t_quantile(.975,1),12.7062047,1e-5); assert 0<=t_cdf(2,10)<=1 and 0<=ibeta(.3,2,5)<=1
