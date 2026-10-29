import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import anomaly_detector as ad


class DetectorTests(unittest.TestCase):
    def test_real_dataset_validates_and_is_reproducible(self):
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv"))
        rows = ad.load_validate(p)
        self.assertEqual(len(rows), 10000)
        a = ad.detect(rows)
        b = ad.detect(rows)
        self.assertEqual(a[0], b[0])

    def test_rejects_bad_schema(self):
        with tempfile.NamedTemporaryFile("w", delete=False, newline="") as f:
            f.write("bad\n1\n")
            p = f.name
        try:
            with self.assertRaises(ValueError): ad.load_validate(p)
        finally:
            os.unlink(p)

    def test_report_contains_assessment(self):
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv"))
        report = ad.render(p)
        self.assertIn("assessment:", report)
        self.assertIn("limitations/confounders:", report)


if __name__ == "__main__":
    unittest.main()
