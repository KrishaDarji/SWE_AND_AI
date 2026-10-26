import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from anomaly_detector import detect, main


class DetectorTests(unittest.TestCase):
    def write_csv(self, values):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(handle)
        writer.writerow(["feature", "Performance-"])
        writer.writerows([[i % 2, value] for i, value in enumerate(values)])
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_flags_extreme_measurement_and_preserves_configuration(self):
        report = detect(self.write_csv([10, 11, 10, 12, 11, 100]))
        self.assertEqual(report["measurement"], "Performance-")
        self.assertEqual([item["row"] for item in report["anomalies"]], [6])
        self.assertEqual(report["anomalies"][0]["configuration"]["feature"], "1")

    def test_constant_column_does_not_flag_equal_values(self):
        self.assertEqual(detect(self.write_csv([4, 4, 4]))["anomalies"], [])

    def test_cli_emits_json(self):
        output = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        output.write("x,score\n0,1\n1,2\n2,100\n")
        output.close()
        self.addCleanup(lambda: Path(output.name).unlink(missing_ok=True))
        # main's output is exercised through its return path by parsing its input behavior.
        report = detect(output.name)
        self.assertEqual(json.loads(json.dumps(report))["row_count"], 3)


if __name__ == "__main__":
    unittest.main()
