import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fix", ROOT / "source" / "fix_slow_config.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


class SuggestionTests(unittest.TestCase):
    def test_selects_best_one_flip(self):
        rows = [
            {"a": "0", "b": "0", "Performance-": 10.0},
            {"a": "1", "b": "0", "Performance-": 12.0},
            {"a": "0", "b": "1", "Performance-": 11.0},
            {"a": "1", "b": "1", "Performance-": 20.0},
        ]
        self.assertEqual(fix.suggest(rows, {"a": "0", "b": "0"})["setting"], "a")

    def test_no_improvement(self):
        rows = [{"a": "0", "Performance-": 5.0}, {"a": "1", "Performance-": 4.0}]
        self.assertEqual(fix.suggest(rows, {"a": "0"})["status"], "no_improvement")

    def test_real_dataset_has_recommendation(self):
        path = Path(__file__).resolve().parents[7] / "moot/optimize/config/X264_AllMeasurements.csv"
        result = fix.suggest(fix.read_measurements(path))
        self.assertEqual(result["status"], "improvement")


if __name__ == "__main__":
    unittest.main()
