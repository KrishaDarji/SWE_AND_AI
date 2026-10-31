import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "source"))
from stream_stats import Flow, process, csv_records


class FlowTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(list(process([])), [])

    def test_one(self):
        snap = list(process([{"Performance-": "3", "ref": "1"}]))[0][1]
        self.assertEqual(snap["count"], 1)
        self.assertEqual(snap["mean"], 3)
        self.assertIsNone(snap["sample_variance"])

    def test_prefixes_batch_and_tie(self):
        rows = [{"Performance-": str(x), "cfg": str(x)} for x in [3, 1, 1, 5]]
        for i, (_, snap) in enumerate(process(rows), 1):
            vals = [3, 1, 1, 5][:i]
            self.assertEqual(snap["count"], i)
            self.assertAlmostEqual(snap["mean"], sum(vals) / i)
            var = sum((x - sum(vals) / i) ** 2 for x in vals) / (i - 1) if i > 1 else None
            if var is None: self.assertIsNone(snap["sample_variance"])
            else: self.assertAlmostEqual(snap["sample_variance"], var)
        self.assertEqual(list(process(rows))[-1][1]["best"]["record_id"], 2)

    def test_invalid_outcomes(self):
        rows = [{"Performance-": x} for x in ["?", "nan", "inf", "2"]]
        snaps = list(process(rows))
        self.assertEqual([s["count"] for _, s in snaps], [0, 0, 0, 1])

    def test_copying_and_configuration(self):
        row = {"Performance-": "2", "nested": {"x": 1}}
        snapshots = list(process([row]))
        row["nested"]["x"] = 9
        snapshots[0][1]["best"]["configuration"]["nested"]["x"] = 7
        self.assertEqual(snapshots[0][1]["best"]["configuration"]["nested"]["x"], 7)
        self.assertEqual(row["nested"]["x"], 9)

    def test_csv_integration(self):
        path = os.path.join(ROOT, "fixture.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("ref,Performance-\n1,4\n2,2\n")
        try:
            out = list(process(csv_records(path)))
            self.assertEqual(out[-1][1]["best"]["configuration"]["ref"], "2")
        finally:
            os.remove(path)

    def test_iterator_is_consumed_one_at_a_time(self):
        calls = []
        def records():
            for x in [3, 2]:
                calls.append(x)
                yield {"Performance-": x}
                self.assertEqual(len(calls), x == 2 and 2 or 1)
        list(process(records()))


if __name__ == "__main__": unittest.main()
