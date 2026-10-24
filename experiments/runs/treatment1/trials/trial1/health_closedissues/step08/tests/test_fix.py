import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source.fix import suggest_fix


class FixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.writer(self.tmp)
        writer.writerow(["MRE-", "ACC+", "PRED40+", "priority", "owner"])
        writer.writerows([[.9, .2, .3, 1, "a"], [.4, .7, .8, 2, "a"],
                          [.6, .5, .6, 1, "b"], [.3, .8, .9, 2, "b"]])
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_smallest_improving_change(self):
        result = suggest_fix(self.tmp.name, {"priority": 1, "owner": "a"})
        self.assertTrue(result["improvement"])
        self.assertEqual(result["setting"], "priority")
        self.assertEqual(result["to"], 2)

    def test_defaults_to_worst_baseline(self):
        result = suggest_fix(self.tmp.name)
        self.assertEqual(result["baseline"]["priority"], 1)
        self.assertEqual(result["baseline"]["owner"], "a")

    def test_requires_measured_baseline(self):
        with self.assertRaises(ValueError):
            suggest_fix(self.tmp.name, {"priority": 9, "owner": "z"})


if __name__ == "__main__":
    unittest.main()
