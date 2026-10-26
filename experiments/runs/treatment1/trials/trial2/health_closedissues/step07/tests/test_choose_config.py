import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source" / "choose_config.py"
spec = importlib.util.spec_from_file_location("choose_config", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChooseConfigTests(unittest.TestCase):
    def write_csv(self, rows):
        fields = list(module.REQUIRED)
        f = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def row(self, mre, acc, pred):
        return {"MRE-": mre, "ACC+": acc, "PRED40+": pred,
                "N_estimators": "10", "criterion": "poisson",
                "Min_sample_leaves": "1", "Min_impurity_decrease": "0",
                "Max_depth": "5"}

    def test_balanced_score_selects_joint_quality(self):
        path = self.write_csv([self.row("0.0", "1.0", "0.0"),
                               self.row("0.1", "0.8", "0.9")])
        selected, score = module.choose(path)
        self.assertEqual(selected["ACC+"], "0.8")
        self.assertAlmostEqual(score, 0.86)

    def test_rejects_missing_columns(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            f.write("ACC+\n0.5\n")
            name = f.name
        self.addCleanup(Path(name).unlink)
        with self.assertRaises(ValueError):
            module.choose(name)

    def test_empty_file(self):
        path = self.write_csv([])
        with self.assertRaises(ValueError):
            module.choose(path)


if __name__ == "__main__":
    unittest.main()
