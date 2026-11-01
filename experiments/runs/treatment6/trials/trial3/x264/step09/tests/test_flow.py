import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class FlowTests(unittest.TestCase):
    def test_prefixes_and_earliest_tie(self):
        rows = [{"mode":"a", "Performance-": v} for v in ["4", "?", "0", "nan", "2", "0"]]
        got = list(summary.flow_rows(iter(rows)))
        self.assertEqual([x["total_rows"] for x in got], list(range(1, 7)))
        self.assertEqual(got[-1]["valid_count"], 4)
        self.assertEqual(got[-1]["missing_count"], 1)
        self.assertEqual(got[-1]["invalid_count"], 1)
        self.assertAlmostEqual(got[-1]["mean"], 1.5)
        self.assertAlmostEqual(got[-1]["sample_variance"], 11/3)
        self.assertEqual(got[-1]["best"], {"value": 0.0, "position": 3, "settings": {"mode":"a"}})

    def test_empty_single_zero_and_snapshot_independence(self):
        self.assertEqual(summary.flow_final(self._csv("mode,Performance-\n"))["total_rows"], 0)
        snaps = list(summary.flow_rows(iter([{"mode":"x", "Performance-":"0"}])))
        self.assertEqual(snaps[0]["mean"], 0.0)
        self.assertIsNone(snaps[0]["sample_variance"])
        first = list(summary.flow_rows(iter([{"mode":"x", "Performance-":"1"}, {"mode":"y", "Performance-":"2"}])))[0]
        self.assertEqual(first["total_rows"], 1)
        self.assertIsNone(first["best"]["settings"].get("later"))

    def test_csv_integration_is_one_pass(self):
        path = self._csv("a,b,Performance-\nA,1,3\nB,2,1\n")
        values = list(summary.flow_csv(path))
        self.assertEqual(values[-1]["best"]["position"], 2)
        self.assertEqual(values[-1]["best"]["settings"], {"a":"B", "b":"2"})

    def _csv(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: Path(f.name).unlink())
        return f.name

if __name__ == "__main__": unittest.main()
