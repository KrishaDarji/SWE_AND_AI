import unittest

from source.justify_estimate import TARGET, build_model, load_data, make_report


DATASET = "/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv"


class JustificationTests(unittest.TestCase):
    def test_loads_target_and_rows(self):
        x, y, numeric, categorical = load_data(DATASET)
        self.assertGreater(len(y), 20)
        self.assertEqual(len(x), len(y))
        self.assertIn("Kloc+", numeric)
        self.assertIn("rely", categorical)

    def test_report_contains_decision_evidence(self):
        report = make_report(DATASET)
        for phrase in ("Estimate:", "planning range:", "cross-validation", "Recommendation:", "Risk:"):
            self.assertIn(phrase, report)

    def test_model_predicts_finite_value(self):
        x, y, numeric, categorical = load_data(DATASET)
        model = build_model(numeric, categorical).fit(x, y)
        value = float(model.predict([x[0]])[0])
        self.assertTrue(value == value)
        self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
