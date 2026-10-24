import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "source" / "choose.py"
spec = importlib.util.spec_from_file_location("choose", SOURCE)
choose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choose)


class ChooseTests(unittest.TestCase):
    def write_csv(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=["feature", "Performance-"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_selects_lowest_eligible_measurement(self):
        path = self.write_csv([
            {"feature": "a", "Performance-": "20"},
            {"feature": "b", "Performance-": "10.5"},
            {"feature": "c", "Performance-": "5"},
        ])
        config, performance = choose.choose_configuration(path, 10.5)
        self.assertEqual(config, {"feature": "c"})
        self.assertEqual(performance, 5.0)

    def test_rejects_empty_feasible_set(self):
        path = self.write_csv([{"feature": "a", "Performance-": "20"}])
        with self.assertRaises(ValueError):
            choose.choose_configuration(path, 19)


if __name__ == "__main__":
    unittest.main()
