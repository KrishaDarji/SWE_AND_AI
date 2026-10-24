import csv
import pathlib
import tempfile
import unittest

from source.anomaly_detector import analyze


class AnomalyDetectorTests(unittest.TestCase):
    def make_csv(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=["idX", "Effort-"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        self.addCleanup(pathlib.Path(handle.name).unlink)
        return handle.name

    def test_flags_extreme_effort(self):
        report = analyze(self.make_csv([{"idX": i, "Effort-": value}
                                        for i, value in enumerate([10, 11, 10, 9, 10, 1000], 1)]))
        self.assertEqual([item["id"] for item in report["outliers"]], ["6"])

    def test_reports_invalid_and_duplicate_ids(self):
        report = analyze(self.make_csv([
            {"idX": "1", "Effort-": "10"},
            {"idX": "1", "Effort-": "bad"},
            {"idX": "2", "Effort-": "11"},
        ]))
        self.assertEqual(report["invalid_rows"], [3])
        self.assertEqual(report["duplicate_ids"], {"1": [2, 3]})

    def test_rejects_missing_target(self):
        with self.assertRaises(ValueError):
            analyze(self.make_csv([{ "idX": "1", "Effort-": "10" }]), "Missing")


if __name__ == "__main__":
    unittest.main()
