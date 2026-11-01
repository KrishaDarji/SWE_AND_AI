import csv, json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_spot import spot

class SpotTest(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        w = csv.writer(f); w.writerow(["a", "b", "Performance-"]); w.writerows(rows); f.close(); self.addCleanup(lambda: Path(f.name).unlink()); return f.name
    def test_normal_and_anomalous(self):
        result = spot(self.write([[0, "x", 10], [0, "y", 11], [1, "x", 9], [1, "y", 10], [0, "z", 100]]))
        self.assertEqual(result["flagged_record_count"], 1); self.assertEqual(result["records"][-1]["comparison_group_count"], 2)
    def test_isolation_and_duplicate_exclusion(self):
        result = spot(self.write([[0, "x", 10], [0, "x", 99], [1, "x", 10], [1, "y", 11], [1, "z", 12]]))
        first = result["records"][0]; self.assertNotIn(2, first["comparison_record_ids"]); self.assertNotIn(1, first["comparison_record_ids"])
    def test_small_empty_constant_and_json(self):
        for rows in ([], [[0, "x", 1]], [[0, "x", 1], [1, "x", 1]]):
            result = spot(self.write(rows)); json.dumps(result, allow_nan=False)
        self.assertIsNone(spot(self.write([[0, "x", 1], [1, "x", 1]]))["records"][0]["score"])

if __name__ == "__main__": unittest.main()
