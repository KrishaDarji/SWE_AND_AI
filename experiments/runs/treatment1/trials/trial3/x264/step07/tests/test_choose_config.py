import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source" / "choose_config.py"
SPEC = importlib.util.spec_from_file_location("choose_config", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ChooseConfigTests(unittest.TestCase):
    def write_csv(self, rows):
        file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        self.addCleanup(lambda: Path(file.name).unlink(missing_ok=True))
        writer = csv.DictWriter(file, fieldnames=["flag", "Performance-"])
        writer.writeheader()
        writer.writerows(rows)
        file.close()
        return file.name

    def test_selects_lowest_feasible_measurement(self):
        path = self.write_csv([
            {"flag": "a", "Performance-": "10"},
            {"flag": "b", "Performance-": "3.5"},
            {"flag": "c", "Performance-": "2"},
        ])
        result = MODULE.choose_configuration(path, 3.5)
        self.assertEqual(result["performance"], 2.0)
        self.assertEqual(result["configuration"]["flag"], "c")

    def test_rejects_empty_feasible_set(self):
        path = self.write_csv([{"flag": "a", "Performance-": "10"}])
        with self.assertRaisesRegex(ValueError, "no configuration"):
            MODULE.choose_configuration(path, 9)

    def test_requires_measurement_column(self):
        path = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.addCleanup(lambda: Path(path.name).unlink(missing_ok=True))
        path.write("flag\na\n")
        path.close()
        with self.assertRaisesRegex(ValueError, "Performance-"):
            MODULE.choose_configuration(path.name, 10)


if __name__ == "__main__":
    unittest.main()
