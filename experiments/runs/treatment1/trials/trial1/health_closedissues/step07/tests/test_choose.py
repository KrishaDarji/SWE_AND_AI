import csv
import importlib.util
import pathlib
import tempfile
import unittest


SOURCE = pathlib.Path(__file__).parents[1] / "source" / "choose.py"
spec = importlib.util.spec_from_file_location("choose", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChooseTests(unittest.TestCase):
    def make_csv(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=["MRE-", "ACC+", "model"])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        self.addCleanup(lambda: pathlib.Path(handle.name).unlink())
        return handle.name

    def test_accuracy_constraint_maximizes_accuracy_then_error(self):
        path = self.make_csv([
            {"MRE-": "0.2", "ACC+": "0.8", "model": "a"},
            {"MRE-": "0.1", "ACC+": "0.9", "model": "b"},
            {"MRE-": "0.3", "ACC+": "0.9", "model": "c"},
        ])
        row, accuracy, error = module.choose_configuration(path, min_accuracy=0.8)
        self.assertEqual(row["model"], "b")
        self.assertEqual((accuracy, error), (0.9, 0.1))

    def test_error_constraint_minimizes_error_then_accuracy(self):
        path = self.make_csv([
            {"MRE-": "0.2", "ACC+": "0.8", "model": "a"},
            {"MRE-": "0.1", "ACC+": "0.7", "model": "b"},
            {"MRE-": "0.1", "ACC+": "0.9", "model": "c"},
        ])
        row, _, _ = module.choose_configuration(path, max_error=0.2)
        self.assertEqual(row["model"], "c")

    def test_requires_an_eligible_configuration(self):
        path = self.make_csv([{"MRE-": "0.2", "ACC+": "0.8", "model": "a"}])
        with self.assertRaises(ValueError):
            module.choose_configuration(path, min_accuracy=0.9)


if __name__ == "__main__":
    unittest.main()
