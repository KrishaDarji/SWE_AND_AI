import csv, os, tempfile, unittest
from source.x264_justify import justify

class JustifyTests(unittest.TestCase):
    def fixture(self, rows):
        h = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        w = csv.DictWriter(h, fieldnames=["a", "b", "Performance-"])
        w.writeheader(); w.writerows(rows); h.close(); self.addCleanup(lambda: os.unlink(h.name)); return h.name

    def test_selection_tie_reference_and_complete_configuration(self):
        r = justify(self.fixture([{"a":"x","b":"0","Performance-":"10"}, {"a":"y","b":"1","Performance-":"4"}, {"a":"z","b":"2","Performance-":"4"}]))
        self.assertEqual(r["recommendation"]["record_id"], 1)
        self.assertEqual(r["recommendation"]["configuration"], {"a":"y","b":"1"})
        self.assertEqual(r["historical_reference"]["record_id"], 0)
        self.assertEqual(r["comparison"]["absolute_difference"], 6)
        self.assertEqual(r["comparison"]["percentage_improvement"], 60)
        self.assertIn("Associations", " ".join(r["limitations"]))

    def test_missing_nonfinite_invalid_and_zero_reference(self):
        r = justify(self.fixture([{"a":"x","b":"0","Performance-":"0"}, {"a":"y","b":"1","Performance-":"nan"}, {"a":"z","b":"2","Performance-":"-2"}]))
        self.assertEqual(r["recommendation"]["outcome"], -2)
        self.assertEqual(r["comparison"]["status"], "zero_reference")
        self.assertIsNone(r["comparison"]["percentage_improvement"])

    def test_no_usable_records_and_missing_reference_outcome(self):
        with self.assertRaises(ValueError): justify(self.fixture([{"a":"x","b":"0","Performance-":"?"}]))
        r = justify(self.fixture([{"a":"x","b":"0","Performance-":"bad"}, {"a":"y","b":"1","Performance-":"2"}]))
        self.assertEqual(r["comparison"]["status"], "unavailable")

if __name__ == "__main__": unittest.main()
