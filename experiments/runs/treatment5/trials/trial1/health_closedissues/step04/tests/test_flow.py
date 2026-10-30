import csv, math, os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "source"))
from flow import CONFIG, OUTCOMES, process_rows, run

def row(i, a, b, c):
    return {"MRE-": a, "ACC+": b, "PRED40+": c, **dict(zip(CONFIG, [i, "criterion", i, i / 10, i]))}

class FlowTests(unittest.TestCase):
    def test_empty_and_one(self):
        self.assertEqual(list(process_rows([])), [])
        s = list(process_rows(iter([row(1, "2", "3", "4")])))[0][1]
        for n in OUTCOMES:
            self.assertEqual(s[n]["valid_count"], 1); self.assertIsNone(s[n]["sample_variance"])
    def test_directions_ties_and_batch(self):
        data = [row(1, 3, 2, 4), row(2, 1, 5, 2), row(3, 1, 5, 9)]
        snapshots = list(process_rows(iter(data))); final = snapshots[-1][1]
        self.assertEqual(final["MRE-"]["best_value"], 1.0); self.assertEqual(final["MRE-"]["best_record_id"], 2)
        self.assertEqual(final["ACC+"]["best_record_id"], 2); self.assertEqual(final["PRED40+"]["best_record_id"], 3)
        self.assertAlmostEqual(final["MRE-"]["mean"], 5 / 3)
        self.assertAlmostEqual(final["ACC+"]["sample_variance"], 3.0)
    def test_missing_nonfinite_and_defensive_copy(self):
        data = [row(1, "?", "nan", "inf"), row(2, 4, 6, 8)]
        snaps = list(process_rows(iter(data)))
        self.assertEqual(snaps[0][1]["MRE-"]["valid_count"], 0)
        snaps[1][1]["MRE-"]["best_config"][0] = "mutated"
        self.assertEqual(snaps[0][1]["MRE-"]["best_config"], None)
    def test_iterator_and_csv_integration(self):
        used = []
        def source():
            for x in [row(1, 1, 2, 3), row(2, 4, 5, 6)]: used.append(x); yield x
        self.assertEqual(len(list(process_rows(source()))), 2); self.assertEqual(len(used), 2)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[*OUTCOMES, *CONFIG]); w.writeheader(); w.writerow(row(1,1,2,3)); w.writerow(row(2,4,5,6))
            snaps, final = run(p)
            self.assertEqual(len(snaps), 2); self.assertEqual(final["PRED40+"]["best_value"], 6.0)

if __name__ == "__main__": unittest.main()
