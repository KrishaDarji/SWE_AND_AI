import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("justify", ROOT / "source" / "justify.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JustifyTests(unittest.TestCase):
    def make_csv(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "Performance-"])
        writer.writerows(rows)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_selects_highest_mean_and_renders_reason(self):
        path = self.make_csv([[0, 0, 10], [0, 0, 12], [1, 0, 20], [1, 0, 22], [0, 1, 19]])
        columns, key, summary, count, rank = MODULE.recommend(path)
        self.assertEqual(columns, ["a", "b"])
        self.assertEqual(key, (1, 0))
        self.assertAlmostEqual(summary.mean, 21.0)
        self.assertEqual((count, rank), (3, 1))
        self.assertIn("rank 1/3", MODULE.render(path))

    def test_rejects_non_binary_configuration(self):
        path = self.make_csv([[2, 0, 1]])
        with self.assertRaises(ValueError):
            MODULE.load_measurements(path)

    def test_rejects_wrong_width(self):
        path = self.make_csv([[0, 0]])
        with self.assertRaises(ValueError):
            MODULE.load_measurements(path)


if __name__ == "__main__":
    unittest.main()
