"""Behavioral tests using local fixtures and the supplied historical dataset."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


STEP = Path(__file__).resolve().parents[1]
SOURCE = STEP / "source" / "remember.py"
sys.path.insert(0, str(SOURCE.parent))
import remember


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(dir=STEP / "tests")
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)

    def dataset(self, text):
        path = self.directory / "input.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_statistics_rankings_and_distributions(self):
        path = self.dataset("flag,constant,Performance-\n0,1,4\n1,1,2\n0,1,2\n1,1,8\n")
        result = remember.summarize(remember.load_measurements(path), 3)
        stats = result["objective"]["statistics"]
        self.assertEqual((stats["mean"], stats["median"], stats["q25"], stats["q75"]),
                         (4, 3, 2, 5))
        self.assertAlmostEqual(stats["population_stddev"], 6 ** 0.5)
        self.assertEqual([row["data_row"] for row in result["top_runs"]], [2, 3, 1])
        self.assertEqual([row["data_row"] for row in result["best_runs"]], [2, 3])
        self.assertEqual(result["unique_configurations"], 2)
        self.assertEqual(result["repeated_configuration_rows"], 2)
        self.assertTrue(result["settings"]["constant"]["constant"])
        group = result["settings"]["flag"]["values"][0]
        self.assertEqual((group["count"], group["fraction"], group["objective"]["mean"]),
                         (2, 0.5, 3))
        self.assertEqual(result["quality_measures"], [])

    def test_objective_position_whitespace_and_duplicates(self):
        path = self.dataset("Performance-,flag\n 3.5 ,1\n3.5,1\n")
        result = remember.summarize(remember.load_measurements(path))
        self.assertEqual(result["duplicate_measurement_rows"], 1)
        self.assertEqual(result["objective"]["statistics"]["population_stddev"], 0)
        self.assertEqual(result["top_runs"][0]["settings"], {"flag": 1})

    def test_rejects_invalid_csv(self):
        cases = ["", "a,a,Performance-\n1,1,2\n", "a\n1\n",
                 "a,Performance-\n", "a,Performance-\n1\n",
                 "a,Performance-\n1,2,3\n", "a,Performance-\n,2\n",
                 "a,Performance-\n1,nan\n", "a,Performance-\ninf,2\n",
                 "a,Performance-\nx,2\n", ",Performance-\n1,2\n",
                 "Performance-\n2\n"]
        for text in cases:
            with self.subTest(csv=text), self.assertRaises(ValueError):
                remember.load_measurements(self.dataset(text))

    def test_single_row_and_invalid_top(self):
        data = remember.load_measurements(self.dataset("flag,Performance-\n1,7\n"))
        self.assertEqual(remember.summarize(data)["objective"]["statistics"]["q25"], 7)
        with self.assertRaises(ValueError):
            remember.summarize(data, 0)

    def test_cli_writes_valid_reports_and_preserves_input(self):
        path = self.dataset("flag,Performance-\n0,9\n1,2\n")
        before = path.read_bytes()
        output = self.directory / "reports"
        process = subprocess.run([sys.executable, str(SOURCE), str(path),
                                  "--output-dir", str(output), "--top", "1"],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["top_runs"][0]["Performance-"], 2)
        self.assertEqual(len(report["top_runs"]), 1)
        self.assertEqual((output / "summary.txt").read_text(), process.stdout)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_failure_is_actionable(self):
        path = self.dataset("flag,Performance-\n1,nan\n")
        process = subprocess.run([sys.executable, str(SOURCE), str(path),
                                  "--output-dir", str(self.directory / "reports")],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn("nonfinite", process.stderr)
        self.assertFalse((self.directory / "reports").exists())

    def test_cli_refuses_to_overwrite_dataset(self):
        path = self.directory / "summary.json"
        path.write_text("flag,Performance-\n1,2\n")
        before = path.read_bytes()
        with self.assertRaises(SystemExit) as result:
            remember.main([str(path), "--output-dir", str(self.directory)])
        self.assertEqual(result.exception.code, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_actual_dataset_matches_independent_csv_calculations(self):
        path = Path(os.environ["X264_DATASET"])
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        result = remember.summarize(remember.load_measurements(path))
        objectives = [float(row["Performance-"]) for row in rows]
        self.assertEqual(result["row_count"], 1152)
        self.assertEqual(result["setting_count"], 16)
        self.assertEqual(set(result["columns"]), set(columns))
        self.assertEqual(result["objective"]["statistics"]["min"], min(objectives))
        self.assertAlmostEqual(result["objective"]["statistics"]["mean"],
                               sum(objectives) / len(objectives))
        for name, setting in result["settings"].items():
            self.assertTrue(setting["binary"])
            self.assertEqual(sum(group["count"] for group in setting["values"]), len(rows))
            for group in setting["values"]:
                expected = [float(row["Performance-"]) for row in rows
                            if float(row[name]) == group["value"]]
                self.assertEqual(group["count"], len(expected))
                self.assertAlmostEqual(group["objective"]["mean"], sum(expected) / len(expected))


if __name__ == "__main__":
    unittest.main()
