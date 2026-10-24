import csv
import pathlib
import tempfile
import unittest

from source.health_closedissues import detect


class DetectorTests(unittest.TestCase):
    def write(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=["MRE-", "ACC+", "n_estimators"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        self.addCleanup(pathlib.Path(handle.name).unlink)
        return handle.name

    def test_flags_extreme_measurement(self):
        path = self.write([{"MRE-": x, "ACC+": .5, "n_estimators": 10} for x in [1, 1.1, 1, .9, 1, 100]])
        report = detect(path)
        self.assertEqual([x["row"] for x in report["columns"]["MRE-"]["outliers"]], [7])

    def test_reports_invalid_measurements(self):
        report = detect(self.write([{"MRE-": "bad", "ACC+": .5, "n_estimators": 10},
                                    {"MRE-": 1, "ACC+": .6, "n_estimators": 10}]))
        self.assertEqual(report["columns"]["MRE-"]["invalid_rows"], [2])

    def test_ignores_configuration_column(self):
        report = detect(self.write([{"MRE-": 1, "ACC+": .5, "n_estimators": 1000}]))
        self.assertNotIn("n_estimators", report["columns"])


if __name__ == "__main__":
    unittest.main()
