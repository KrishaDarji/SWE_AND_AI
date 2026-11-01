import math,unittest
from source.analyzer import flow,FlowEvaluator

def row(pos,mre,acc,pred,criterion="c",n=1):
    return pos,{"MRE-":mre,"ACC+":acc,"PRED40+":pred,"N_estimators":str(n),"criterion":criterion,"Min_sample_leaves":"2","Min_impurity_decrease":"0.1","Max_depth":"3"}

class FlowTests(unittest.TestCase):
    def test_prefixes_independent_stats_and_directions(self):
        data=[row(2,3,1,5,n=1),row(3,1,4,2,n=2),row(4,1,0,9,n=3)]
        got=flow(data)
        self.assertEqual([x["total_rows_seen"] for x in got],[1,2,3])
        self.assertEqual(got[1]["outcomes"]["MRE-"]["finite_count"],2)
        self.assertAlmostEqual(got[1]["outcomes"]["MRE-"]["mean"],2)
        self.assertAlmostEqual(got[1]["outcomes"]["MRE-"]["sample_variance"],2)
        self.assertEqual(got[-1]["best"]["MRE-"]["position"],3)
        self.assertEqual(got[-1]["best"]["ACC+"]["position"],3)
        self.assertEqual(got[-1]["best"]["PRED40+"]["position"],4)
        got[-1]["outcomes"]["MRE-"]["mean"]=999
        self.assertEqual(got[1]["outcomes"]["MRE-"]["mean"],2)
    def test_empty_single_missing_zero_and_configuration(self):
        self.assertEqual(flow([])[0:] , [])
        x=flow([row(8,0,0,0,"?",7),row(9,"nan",None,"?", "c",8)])[0]
        self.assertEqual(x["outcomes"]["MRE-"]["mean"],0); self.assertIsNone(x["outcomes"]["ACC+"]["sample_variance"])
        self.assertEqual(x["best"]["MRE-"]["configuration"]["N_estimators"],7)
        self.assertIsNone(x["best"]["MRE-"]["configuration"]["criterion"])
    def test_one_pass_iterator(self):
        class Once:
            def __init__(self):self.used=0
            def __iter__(self):
                self.used+=1
                if self.used>1:raise AssertionError("iterated twice")
                yield row(2,1,2,3)
        source=Once(); self.assertEqual(FlowEvaluator().update(*next(iter(source)))["total_rows_seen"],1); self.assertEqual(source.used,1)
    def test_malformed_csv_is_skipped_by_validation_rows(self):
        import tempfile,os
        from source.analyzer import flow_dataset
        f=tempfile.NamedTemporaryFile(mode="w",delete=False); f.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n1,2,3,1,c,2,0,3\n1,2\n\n"); f.close(); self.addCleanup(lambda:os.unlink(f.name))
        self.assertEqual(flow_dataset(f.name)["total_rows_seen"],1)
if __name__=="__main__":unittest.main()
