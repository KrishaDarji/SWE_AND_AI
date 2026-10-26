import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import justify_estimate


DATASET = pathlib.Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"


class EstimateTests(unittest.TestCase):
    def test_dataset_loads_and_has_expected_population(self):
        rows = justify_estimate.load_rows(DATASET)
        self.assertEqual(len(rows), 93)

    def test_estimate_has_point_and_ordered_band(self):
        result = justify_estimate.justify(DATASET, 20)
        self.assertGreater(result["effort_person_months"], 0)
        self.assertLess(result["effort_80_percent_low"], result["effort_person_months"])
        self.assertLess(result["effort_person_months"], result["effort_80_percent_high"])
        self.assertIn("historical NASA projects", result["justification"])

    def test_invalid_schema_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("size,effort\n1,2\n")
            name = f.name
        try:
            with self.assertRaises(ValueError):
                justify_estimate.load_rows(name)
        finally:
            pathlib.Path(name).unlink()


if __name__ == "__main__":
    unittest.main()
