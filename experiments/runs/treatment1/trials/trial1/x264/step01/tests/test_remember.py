import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from remember import summarize  # noqa: E402


class RememberTests(unittest.TestCase):
    def make_csv(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_counts_and_ranks_lower_measure_first(self):
        path = self.make_csv("a,b,Performance-\n1,0,3\n0,1,1\n1,1,2\n")
        result = summarize(path, top=2)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["measure_summary"]["median"], 2.0)
        self.assertEqual([item["measure"] for item in result["best_runs"]], [1.0, 2.0])
        self.assertEqual(result["setting_summary"]["a"]["counts"], {"0": 1, "1": 2})

    def test_real_dataset_schema_and_json_serializable(self):
        root = Path(__file__).resolve().parents[8]
        dataset = root / "moot/optimize/config/X264_AllMeasurements.csv"
        result = summarize(dataset, top=3)
        self.assertGreater(result["rows"], 0)
        self.assertEqual(result["measure"], "Performance-")
        json.dumps(result)

    def test_missing_measure_fails_clearly(self):
        path = self.make_csv("a,b\n1,2\n")
        with self.assertRaisesRegex(ValueError, "measure column"):
            summarize(path)


if __name__ == "__main__":
    unittest.main()
