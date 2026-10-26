import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from remember import summarize


class RememberTests(unittest.TestCase):
    def csv_file(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                             newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_statistics_and_order(self):
        path = self.csv_file("preset,mode,Performance-\na,x,3\nb,y,1\na,y,2\n")
        result = summarize(path, top=2)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["measure_summary"]["median"], 2.0)
        self.assertEqual([x["measure"] for x in result["best_runs"]], [1.0, 2.0])
        self.assertEqual(result["setting_summary"]["preset"]["counts"], {"a": 2, "b": 1})

    def test_real_dataset_is_serializable(self):
        dataset = Path(__file__).resolve().parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        result = summarize(dataset, top=3)
        self.assertEqual(result["rows"], 1152)
        self.assertEqual(len(result["settings"]), 16)
        json.dumps(result)

    def test_invalid_measure_and_top(self):
        path = self.csv_file("a,Performance-\n1,nope\n")
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            summarize(path)
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize(path, top=0)


if __name__ == "__main__":
    unittest.main()
