import math, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../source"))
from flow import Flow, consume, csv_rows, run_csv

FIELDS = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth"
def row(a,b,c,i,criterion="x",leaf="2",dec="0.1",depth="4"):
    return {k: v for k,v in zip(FIELDS.split(','), map(str,(a,b,c,i,criterion,leaf,dec,depth)))}

class FlowTests(unittest.TestCase):
    def test_prefixes_against_independent_calculation_and_ties(self):
        rows = [(2,row(3,1,5,10)), (3,row(1,4,2,11)), (4,row(1,2,9,12)), (5,row(float('nan'),0,0,13))]
        snapshots = list(consume(iter(rows)))
        for n, snap in enumerate(snapshots, 1):
            for idx, name in enumerate(("MRE-", "ACC+", "PRED40+")):
                values = [float(r[1][name]) for r in rows[:n] if math.isfinite(float(r[1][name]))]
                self.assertEqual(snap["outcomes"][name]["count"], len(values))
                self.assertAlmostEqual(snap["outcomes"][name]["mean"], sum(values)/len(values))
                variance = sum((x-sum(values)/len(values))**2 for x in values)/(len(values)-1) if len(values)>1 else None
                self.assertEqual(snap["outcomes"][name]["sample_variance"], variance)
        self.assertEqual(snapshots[2]["best"]["MRE-"]["position"], 3)
        self.assertEqual(snapshots[-1]["best"]["ACC+"]["position"], 3)
        self.assertEqual(snapshots[-1]["best"]["PRED40+"]["position"], 4)
        self.assertEqual(snapshots[0]["rows_seen"], 1)
        self.assertEqual(snapshots[0]["best"]["MRE-"]["position"], 2)

    def test_empty_single_missing_zero_and_configuration_retention(self):
        f = Flow(); empty = f.snapshot(); self.assertEqual(empty["rows_seen"], 0)
        snap = f.update(7, {"MRE-":"0", "ACC+":"NA", "PRED40+":"inf", "criterion":"z"})
        self.assertEqual(snap["outcomes"]["MRE-"]["mean"], 0.0); self.assertIsNone(snap["outcomes"]["ACC+"]["mean"])
        self.assertEqual(snap["best"]["MRE-"]["configuration"]["criterion"], "z")
        self.assertIsNone(snap["best"]["MRE-"]["configuration"]["N_estimators"])

    def test_one_pass_lazy_and_csv_malformed(self):
        class Once:
            def __init__(self): self.used = 0
            def __iter__(self):
                if self.used: raise AssertionError("iterated twice")
                self.used += 1; yield 2, row(1,2,3,1)
        self.assertEqual(list(consume(Once()))[-1]["rows_seen"], 1)
        handle = tempfile.NamedTemporaryFile(mode="w", delete=False, newline="")
        handle.write(FIELDS+"\n1,2,3,1,x,2,0.1,4\nmalformed\n"); handle.close()
        self.assertEqual(run_csv(handle.name)["rows_seen"], 2); os.unlink(handle.name)

if __name__ == "__main__": unittest.main()
