import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "source" / "fix_slow_config.py"
SPEC = importlib.util.spec_from_file_location("fix_slow_config", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixTests(unittest.TestCase):
    def write_dataset(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=["a", "b", "Performance-"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_picks_best_one_toggle(self):
        path = self.write_dataset([
            {"a": "0", "b": "0", "Performance-": "10"},
            {"a": "1", "b": "0", "Performance-": "7"},
            {"a": "0", "b": "1", "Performance-": "8"},
            {"a": "1", "b": "1", "Performance-": "3"},
        ])
        result = MODULE.suggest(path)
        self.assertEqual(result["target_row"], 0)
        self.assertEqual(result["setting_change"]["name"], "a")
        self.assertEqual(result["setting_change"]["from"], "0")
        self.assertEqual(result["setting_change"]["to"], "1")
        self.assertEqual(result["improved_performance"], 7.0)

    def test_requested_row(self):
        path = self.write_dataset([
            {"a": "0", "b": "0", "Performance-": "10"},
            {"a": "1", "b": "0", "Performance-": "7"},
        ])
        result = MODULE.suggest(path, 0)
        self.assertEqual(result["target_row"], 0)
        self.assertEqual(result["setting_change"]["name"], "a")


if __name__ == "__main__":
    unittest.main()
