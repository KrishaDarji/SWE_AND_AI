import csv
import os
import tempfile
import unittest
from source.x264_flow import Flow, flow, records

class FlowTests(unittest.TestCase):
    def fixture(self, text):
        h = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        h.write(text); h.close(); self.addCleanup(lambda: os.unlink(h.name)); return h.name

    def test_prefixes_match_batch_statistics(self):
        path = self.fixture("a,b,Performance-\n1,x,4\n2,y,2\n3,z,6\n")
        snapshots = flow(path); values = [4.0, 2.0, 6.0]
        for i, snapshot in enumerate(snapshots, 1):
            prefix = values[:i]; mean = sum(prefix) / i
            variance = sum((x - mean) ** 2 for x in prefix) / (i - 1) if i > 1 else 0.0
            self.assertEqual(snapshot["count"], i)
            self.assertAlmostEqual(snapshot["mean"], mean)
            self.assertAlmostEqual(snapshot["sample_variance"], variance)

    def test_empty_one_tie_missing_nonfinite_and_configuration(self):
        self.assertEqual(flow(self.fixture("a,Performance-\n")), [])
        state = Flow(); config = {"a": "x"}
        first = state.update(7, config, "3")
        config["a"] = "mutated"
        self.assertEqual(first["best"]["configuration"], {"a": "x"})
        self.assertEqual(state.update(8, {"a": "tie"}, "3")["best"]["record_id"], 7)
        self.assertEqual(state.update(9, {"a": "bad"}, "?")["count"], 2)
        self.assertEqual(state.update(10, {"a": "nan"}, "NaN")["count"], 2)
        self.assertEqual(state.update(11, {"a": "better"}, "2")["best"]["configuration"], {"a": "better"})

    def test_iterator_requests_next_record_after_processing(self):
        class Guard:
            def __init__(self): self.used = 0
            def __iter__(self):
                for value in [(0, {"a": "x"}, "2"), (1, {"a": "y"}, "1")]:
                    self.used += 1
                    yield value
        state = Flow(); source = Guard()
        for record_id, config, outcome in iter(source):
            state.update(record_id, config, outcome)
            self.assertEqual(source.used, record_id + 1)

if __name__ == "__main__": unittest.main()
