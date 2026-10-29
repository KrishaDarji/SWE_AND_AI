import csv
import unittest
from pathlib import Path

from source.certify_health_closedissues import DATASET, load_validate, quality_scores, run


class CertifyTests(unittest.TestCase):
    def test_validation_and_expected_shape(self):
        rows = load_validate(DATASET)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(len(quality_scores(rows)), len(rows))
        self.assertEqual({r["criterion"] for r in rows}, {"absolute_error", "poisson", "squared_error"})

    def test_report_is_reproducible_and_contains_required_statistics(self):
        first, second = run(DATASET), run(DATASET)
        self.assertEqual(first, second)
        for token in ("effect=", "bootstrap_95ci=", "permutation_p_one_sided=", "statistically_credible_at_alpha_0.05=", "limitations="):
            self.assertIn(token, first)


if __name__ == "__main__":
    unittest.main()
