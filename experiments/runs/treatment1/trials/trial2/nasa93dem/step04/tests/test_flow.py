import csv
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from flow import assess, assess_stream, baseline, load_proposals


DATASET = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class FlowTests(unittest.TestCase):
    def test_loads_dataset_and_assesses_every_proposal(self):
        rows = load_proposals(DATASET)
        results = list(assess_stream(rows))
        self.assertEqual(len(rows), 93)
        self.assertEqual(len(results), len(rows))
        self.assertTrue(all(item["decision"] in {"prioritize", "review", "defer"} for item in results))

    def test_high_risk_proposal_is_deferred(self):
        rows = load_proposals(DATASET)
        stats = baseline(rows)
        row = dict(rows[0])
        row.update({"Effort-": "10000", "Defects-": "10000", "MonthsX": "1000"})
        result = assess(row, stats)
        self.assertEqual(result["decision"], "defer")
        self.assertGreaterEqual(result["risk_score"], 65)

    def test_baseline_is_numeric(self):
        stats = baseline(load_proposals(DATASET))
        self.assertGreater(stats["Effort-"]["median"], 0)
        self.assertGreater(stats["MonthsX"]["p75"], 0)


if __name__ == "__main__":
    unittest.main()
