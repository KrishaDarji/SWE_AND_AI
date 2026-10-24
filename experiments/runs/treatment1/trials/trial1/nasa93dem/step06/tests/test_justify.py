import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify import analyze, load_records, render


class JustifyTests(unittest.TestCase):
    def test_analysis_is_positive_and_uses_all_records(self):
        result = analyze([(10, 50), (20, 120), (40, 260)], 15)
        self.assertEqual(result["records"], 3)
        self.assertGreater(result["recommended_effort"], 0)

    def test_loader_skips_invalid_rows(self):
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as f:
            writer = csv.DictWriter(f, fieldnames=["Kloc+", "Effort-"])
            writer.writeheader()
            writer.writerow({"Kloc+": "2", "Effort-": "10"})
            writer.writerow({"Kloc+": "bad", "Effort-": "10"})
            f.flush()
            self.assertEqual(load_records(f.name), [(2.0, 10.0)])

    def test_render_contains_pm_decision(self):
        text = render(analyze([(10, 50), (20, 100)], 10))
        self.assertIn("Recommended planning estimate", text)
        self.assertIn("NASA93", text)


if __name__ == "__main__":
    unittest.main()
