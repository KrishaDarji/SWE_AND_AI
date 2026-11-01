import csv
import os
import tempfile
import unittest
from source.x264_fix import fix

class FixTests(unittest.TestCase):
    fields = ["a", "b", "c", "Performance-"]
    def fixture(self, rows):
        h = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        w = csv.DictWriter(h, fieldnames=self.fields); w.writeheader(); w.writerows(rows); h.close()
        self.addCleanup(lambda: os.unlink(h.name)); return h.name
    def test_minimum_change_beats_maximum_improvement(self):
        path = self.fixture([{"a":"0","b":"0","c":"0","Performance-":"10"}, {"a":"1","b":"0","c":"0","Performance-":"9"}, {"a":"1","b":"1","c":"1","Performance-":"1"}])
        r = fix(path); self.assertEqual(r["repair"]["record_id"], 1); self.assertEqual(r["repair"]["change_count"], 1)
    def test_numeric_equivalence_tie_and_exact_changes(self):
        path = self.fixture([{"a":"0","b":"2.0","c":"x","Performance-":"10"}, {"a":"1.0","b":"2","c":"x","Performance-":"8"}, {"a":"1","b":"3","c":"x","Performance-":"8"}])
        r = fix(path); self.assertEqual(r["repair"]["record_id"], 1); self.assertEqual(r["repair"]["changed_settings"], [{"setting":"a","old":"0","new":"1.0"}])
    def test_no_improvement_invalid_and_missing_nonfinite(self):
        path = self.fixture([{"a":"0","b":"?","c":"x","Performance-":"10"}, {"a":"0","b":"?","c":"x","Performance-":"nan"}])
        self.assertEqual(fix(path)["status"], "no_repair")
        with self.assertRaises(ValueError): fix(path, 9)
        with self.assertRaises(ValueError): fix(path, -1)
    def test_missing_outcomes_are_ignored_and_cli_baseline_can_be_later(self):
        path = self.fixture([{"a":"0","b":"0","c":"x","Performance-":"bad"}, {"a":"1","b":"0","c":"x","Performance-":"5"}, {"a":"1","b":"1","c":"x","Performance-":"4"}])
        self.assertEqual(fix(path, 1)["repair"]["record_id"], 2)

if __name__ == "__main__":
    unittest.main()
