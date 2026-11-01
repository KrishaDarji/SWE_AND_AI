import csv, os, tempfile, unittest
from source.analyzer import race, _race_classify

H="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def make(body):
    f=tempfile.NamedTemporaryFile("w",delete=False,newline=""); f.write(H+body); f.close(); return f.name
def row(m,a,p,n=1,c="x"): return f"{m},{a},{p},{n},{c},1,0,3\n"
class RaceTests(unittest.TestCase):
    def test_same_population_positions_and_complete_results(self):
        p=make(row(3,5,7,1)+row(1,9,8,101)+row(2,6,6,2)+row(0,0,0,-1)+"?,1,1,1,x,1,0,3\n")
        try:
            x=race(p,100,range(42,52)); self.assertEqual((x["raw_count"],x["parsed_count"],x["feasible_count"]),(5,5,2)); self.assertEqual(x["results"][0]["selector"]["original_position"],4)
            for z in x["results"]: self.assertIn(z["baseline"]["original_position"],[2,4]); self.assertEqual(set(z["baseline"]),set(["original_position","configuration","outcomes"]))
        finally: os.unlink(p)
    def test_reproducible_local_rng_and_summary(self):
        p=make(row(1,1,1)+row(2,2,2,2)+row(3,3,3,3))
        try:
            a=race(p); b=race(p); self.assertEqual(a,b); self.assertEqual(a["summary"]["wins"]+a["summary"]["ties"]+a["summary"]["losses"],10); self.assertAlmostEqual(a["summary"]["mean_MRE_difference"],sum(z["differences"]["MRE- (baseline - selector)"] for z in a["results"])/10)
        finally: os.unlink(p)
    def test_classification_boundaries_and_directions(self):
        self.assertEqual([_race_classify(x,0.1) for x in (0.2,0.1,0,-0.1,-0.2)], ["win","tie","tie","tie","loss"])
    def test_empty_no_feasible_single_and_invalid(self):
        p=tempfile.NamedTemporaryFile("w",delete=False); p.close(); self.assertEqual(race(p.name)["status"],"empty_data"); os.unlink(p.name)
        p=make(row("?",1,1)); self.assertEqual(race(p)["status"],"no_feasible_rows"); os.unlink(p)
        p=make(row(1,2,3)); self.assertEqual(race(p)["feasible_count"],1); os.unlink(p)
        for kwargs in ({"cap":-1},{"tie_tolerance":-1},{"cap":float("inf")},{"tie_tolerance":float("nan")}):
            with self.assertRaises(ValueError): race(make(row(1,2,3)),**kwargs)
    def test_secondary_direction_labels(self):
        p=make(row(1,1,1));
        try: self.assertIn("positive favors selector",race(p)["summary"]["secondary_direction"])
        finally: os.unlink(p)
if __name__=="__main__": unittest.main()
