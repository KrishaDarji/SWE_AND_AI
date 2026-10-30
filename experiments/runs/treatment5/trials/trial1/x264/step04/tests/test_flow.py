import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from flow import Flow, evaluate, csv_records


def batch(rows):
    vals = [float(r["Performance-"]) for r in rows
            if math.isfinite(float(r["Performance-"]))]
    mean = sum(vals) / len(vals) if vals else 0.0
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
    return len(vals), mean, var, min(vals) if vals else None


class FlowTests(unittest.TestCase):
    def test_empty_and_one(self):
        self.assertEqual(evaluate([]), [])
        rows = [{"x": "a", "Performance-": "3"}]
        s = evaluate(rows)[0]
        self.assertEqual((s["count"], s["mean"], s["sample_variance"]), (1, 3.0, 0.0))

    def test_prefixes_match_batch(self):
        rows = [{"x": "a", "Performance-": "3"}, {"x": "b", "Performance-": "1"},
                {"x": "c", "Performance-": "2"}]
        for i, snap in enumerate(evaluate(rows), 1):
            n, mean, var, best = batch(rows[:i])
            self.assertEqual(snap["count"], n)
            self.assertAlmostEqual(snap["mean"], mean)
            self.assertAlmostEqual(snap["sample_variance"], var)
            self.assertEqual(snap["best_performance"], best)

    def test_ties_keep_first_and_configuration(self):
        rows = [{"x": [1], "Performance-": "2"}, {"x": [2], "Performance-": "2"}]
        snaps = evaluate(rows)
        self.assertEqual(snaps[-1]["best_record_id"], 1)
        self.assertEqual(snaps[-1]["best_configuration"], {"x": [1]})

    def test_missing_nonfinite_and_snapshot_copy(self):
        rows = [{"x": "bad"}, {"x": "nan", "Performance-": "nan"},
                {"x": "ok", "Performance-": "4"}]
        snaps = evaluate(rows)
        self.assertEqual(len(snaps), 1)
        first = snaps[0]
        first["best_configuration"]["x"] = "changed"
        self.assertEqual(snaps[0]["best_configuration"]["x"], "changed")
        original = {"x": [1], "Performance-": "1"}
        out = evaluate([original])
        original["x"][0] = 9
        self.assertEqual(out[0]["best_configuration"], {"x": [1]})

    def test_csv_integration_and_lazy_order(self):
        events = []
        class Observable(dict):
            def get(self, key, default=None):
                events.append("processed")
                return super().get(key, default)
        def records():
            yield Observable({"x": "a", "Performance-": "3"})
            self.assertEqual(events, ["processed"])
            yield Observable({"x": "b", "Performance-": "1"})
        self.assertEqual(evaluate(records())[-1]["best_performance"], 1.0)

    def test_csv_reader(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as f:
            f.write("x,Performance-\na,3\n")
            name = f.name
        try:
            self.assertEqual(evaluate(csv_records(name))[-1]["count"], 1)
        finally:
            os.unlink(name)


if __name__ == "__main__":
    unittest.main()
