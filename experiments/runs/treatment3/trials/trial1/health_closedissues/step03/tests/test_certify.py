import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import certify, score_rows


class CertificationTests(unittest.TestCase):
    def test_score_prefers_lower_error_and_higher_positive_metrics(self):
        rows = [{"MRE-": "0", "ACC+": "1", "PRED40+": "1", "criterion": "a"},
                {"MRE-": "1", "ACC+": "0", "PRED40+": "0", "criterion": "b"}]
        scores = score_rows(rows)
        self.assertGreater(scores[0], scores[1])


    def test_certification_finds_clear_winner(self):
        rows = []
        for _ in range(30):
            rows.append({"MRE-": "0.1", "ACC+": "0.9", "PRED40+": "1", "criterion": "good"})
            rows.append({"MRE-": "0.9", "ACC+": "0.2", "PRED40+": "0", "criterion": "bad"})
        report = certify(rows, bootstrap=200, seed=7)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["winner"], "good")
        self.assertTrue(comparison["reliably_better"])
