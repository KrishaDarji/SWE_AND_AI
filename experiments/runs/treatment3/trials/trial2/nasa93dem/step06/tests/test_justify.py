import csv
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("justify", ROOT / "source" / "justify.py")
justify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(justify)

DATA = pathlib.Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class JustifyTests(unittest.TestCase):
    def test_load_and_validate(self):
        rows = justify.load_validate(DATA)
        self.assertEqual(len(rows), 93)
        self.assertIn("Effort-", rows[0])

    def test_representative_proposal_has_all_predictors(self):
        rows = justify.load_validate(DATA)
        proposal = justify.representative_proposal(rows)
        self.assertEqual(set(proposal), set(justify.feature_rows(rows)[0]))

    def test_model_metrics_and_report(self):
        rows = justify.load_validate(DATA)
        model, vectorizer, metrics = justify.model_and_metrics(rows)
        report, estimate, interval = justify.render_report(
            rows, justify.representative_proposal(rows), model, vectorizer, metrics)
        self.assertGreater(estimate, 0)
        self.assertLessEqual(interval[0], estimate)
        self.assertGreater(interval[1], estimate)
        self.assertIn("causal evidence", report)
        self.assertTrue(all(value >= 0 for value in metrics.values() if isinstance(value, float)))


if __name__ == "__main__":
    unittest.main()
