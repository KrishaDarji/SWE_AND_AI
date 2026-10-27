import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import choose_configuration


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "PRED40+", "criterion", "depth"])
        writer.writerows([[1, .9, .8, "a", 2], [2, .8, .7, "a", 3],
                          [8, .2, .1, "b", 2], [9, .1, .2, "b", 3]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_balanced_choice_prefers_good_configuration(self):
        result = choose_configuration(self.file.name, neighbors=1)
        self.assertEqual(result["selection"], "balanced observed and predicted quality")
        self.assertEqual(result["recommended_configuration"]["criterion"], "a")
        self.assertEqual(result["candidates_considered"], 4)

    def test_weight_validation_and_cli(self):
        with self.assertRaises(ValueError):
            choose_configuration(self.file.name, prediction_weight=1.1)
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--choose", "--neighbors", "1"]
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result["recommended_configuration"]["criterion"], "a")


if __name__ == "__main__":
    unittest.main()
