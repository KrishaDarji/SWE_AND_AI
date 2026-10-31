import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from flow_evaluator import FlowEvaluator, csv_records


class FlowTests(unittest.TestCase):
    def test_empty_input(self):
        e = FlowEvaluator()
        self.assertEqual(list(e.process([])), [])
        self.assertEqual(e.stats.count, 0)
        self.assertIsNone(e.best_value)

    def test_one_observation(self):
        s = list(FlowEvaluator().process([{"Performance-": "4", "x": "a"}]))[0]
        self.assertEqual(s["valid_observations"], 1)
        self.assertEqual(s["mean"], 4.0)
        self.assertIsNone(s["sample_variance"])

    def test_prefix_statistics_match_batch(self):
        rows = [{"Performance-": x} for x in [5, 1, 7, 3]]
        e = FlowEvaluator()
        for index, snap in enumerate(e.process(rows), 1):
            values = [float(row["Performance-"]) for row in rows[:index]]
            mean = sum(values) / len(values)
            variance = (sum((x - mean) ** 2 for x in values) /
                        (len(values) - 1) if len(values) > 1 else None)
            self.assertAlmostEqual(snap["mean"], mean)
            if variance is None:
                self.assertIsNone(snap["sample_variance"])
            else:
                self.assertAlmostEqual(snap["sample_variance"], variance)

    def test_tie_keeps_first_best(self):
        snapshots = list(FlowEvaluator().process([
            {"Performance-": 2, "name": "first"},
            {"Performance-": 2, "name": "second"}]))
        self.assertEqual(snapshots[-1]["best_record_id"], 1)
        self.assertEqual(snapshots[-1]["best_configuration"]["name"], "first")

    def test_missing_and_nonfinite_are_skipped(self):
        e = FlowEvaluator()
        snapshots = list(e.process([{}, {"Performance-": "nan"},
                                    {"Performance-": "inf"},
                                    {"Performance-": 8}]))
        self.assertEqual([s["accepted"] for s in snapshots], [False, False, False, True])
        self.assertEqual(snapshots[-1]["valid_observations"], 1)

    def test_full_configuration_and_defensive_snapshots(self):
        row = {"Performance-": 1, "flag": {"value": 9}}
        e = FlowEvaluator()
        first = e.update(row)
        row["flag"]["value"] = 0
        first["best_configuration"]["flag"]["value"] = -1
        second = e.update({"Performance-": 3, "flag": {"value": 4}})
        self.assertEqual(second["best_configuration"]["flag"]["value"], 9)

    def test_csv_streams_before_next_record(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            handle.write("x,Performance-\na,4\nb,2\n")
            path = handle.name
        try:
            records = csv_records(path)
            first = next(records)
            e = FlowEvaluator()
            self.assertEqual(e.update(first)["valid_observations"], 1)
            self.assertEqual(next(records)["x"], "b")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
