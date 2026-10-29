import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from guess_skill import TARGETS, load_dataset, train_guess_model  # noqa: E402


DATASET = Path.cwd() / "moot/optimize/hpo/Health-ClosedIssues0000.csv"


class GuessSkillTests(unittest.TestCase):
    def test_loads_expected_schema_and_row_count(self):
        rows, features, targets = load_dataset(DATASET)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(targets, TARGETS)
        self.assertEqual(features, ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"])

    def test_training_reports_finite_holdout_metrics(self):
        _, metrics = train_guess_model(DATASET)
        self.assertEqual(metrics["test_rows"], 2000)
        for key in ("mae_overall", "rmse_overall", "r2_overall"):
            self.assertTrue(metrics[key] == metrics[key])
        self.assertEqual(set(metrics["per_target"]), set(TARGETS))


if __name__ == "__main__":
    unittest.main()
