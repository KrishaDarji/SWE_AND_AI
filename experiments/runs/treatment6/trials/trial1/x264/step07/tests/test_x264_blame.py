import csv
import json
import os
import tempfile
import unittest

from x264_blame import blame

class BlameTests(unittest.TestCase):
    def write(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        self.addCleanup(lambda: os.unlink(handle.name))
        writer = csv.DictWriter(handle, fieldnames=["strong", "constant", "multi", "correlated", "Performance-"])
        writer.writeheader(); writer.writerows(rows); handle.close(); return handle.name

    def test_known_association_and_deterministic_tie(self):
        path = self.write([
            {"strong":"0","constant":"x","multi":"a","correlated":"0","Performance-":"1"},
            {"strong":"0","constant":"x","multi":"b","correlated":"0","Performance-":"2"},
            {"strong":"1","constant":"x","multi":"a","correlated":"1","Performance-":"9"},
            {"strong":"1","constant":"x","multi":"c","correlated":"1","Performance-":"10"}])
        result = blame(path); by = {x["setting"]: x for x in result["ranking"]}
        self.assertGreater(by["strong"]["score"], by["multi"]["score"])
        self.assertEqual(by["constant"]["score"], 0.0)
        self.assertEqual([x["setting"] for x in result["ranking"]], [x["setting"] for x in blame(path)["ranking"]])
        self.assertEqual(by["strong"]["valid_count"], 4)

    def test_missing_nonfinite_small_groups_and_summaries(self):
        path = self.write([
            {"strong":"0","constant":"x","multi":"a","correlated":"0","Performance-":"1"},
            {"strong":"?","constant":"x","multi":"b","correlated":"0","Performance-":"2"},
            {"strong":"1","constant":"x","multi":"c","correlated":"1","Performance-":"nan"},
            {"strong":"1","constant":"x","multi":"c","correlated":"1","Performance-":"4"}])
        result = blame(path); by = {x["setting"]: x for x in result["ranking"]}
        self.assertEqual(result["rows_with_finite_outcome"], 3)
        self.assertEqual(by["strong"]["missing_count"], 1)
        self.assertEqual(by["multi"]["level_count"], 3)
        self.assertEqual(by["strong"]["levels"][0]["summary"]["count"], 1)

if __name__ == "__main__":
    unittest.main()
