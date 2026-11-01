import unittest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class SpotTests(unittest.TestCase):
    def rows(self, values):
        configs = [(0,0),(0,1),(1,0),(1,1),(0,0),(0,1)]
        return [{"a":str(a),"b":str(b),"Performance-":str(y)} for (a,b),y in zip(configs,values)]

    def test_high_low_flags_and_independent_median_mad(self):
        r = summary.spot_rows(self.rows([10,11,9,10,100,-50]), radius=1, minimum_reference=2, threshold=3.5)
        self.assertEqual([x["position"] for x in r["flags"]], [5,6])
        self.assertEqual(r["flags"][0]["reference_count"], 4)
        self.assertEqual(r["flags"][0]["center"], 9.5)
        self.assertEqual(r["flags"][0]["scale_mad"], 1)

    def test_boundary_tie_duplicate_self_exclusion_and_rows_count(self):
        rows = [{"a":"0","Performance-":"0"},{"a":"0.0","Performance-":"2"},
                {"a":"1","Performance-":"1"},{"a":"1","Performance-":"1"}]
        r = summary.spot_rows(rows, radius=0, minimum_reference=1, threshold=1)
        self.assertEqual(r["flags"], [])
        self.assertEqual(r["eligible_count"], 4)
        self.assertEqual(r["rows_lacking_reference"], [])

    def test_zero_scale_undefined_and_no_reference(self):
        r = summary.spot_rows([{"a":"0","Performance-":"1"},{"a":"1","Performance-":"2"}], radius=0, minimum_reference=1)
        self.assertEqual([x["position"] for x in r["rows_lacking_reference"]], [1,2])
        r = summary.spot_rows([{"a":"0","Performance-":"1"},{"a":"0","Performance-":"1"},{"a":"0","Performance-":"1"}], radius=0, minimum_reference=1, threshold=1)
        self.assertEqual(r["flags"], [])

    def test_invalid_rows_positions_and_full_configuration(self):
        rows = [{"a":"x","b":"0","Performance-":"1"},{"a":"0","b":"0","Performance-":"?"},
                {"a":"0","b":"1","Performance-":"2"}]
        r = summary.spot_rows(rows, minimum_reference=0)
        self.assertEqual([x["position"] for x in r["excluded"]], [1,2])
        self.assertEqual(r["flags"], [])
        self.assertEqual(r["settings"], ["a","b"])

if __name__ == "__main__": unittest.main()
