import os
import tempfile
import unittest

import numpy as np

from source.nasa93_effort import TARGET, build_pipeline, evaluate, predict_proposal, predictor_names
from source.nasa93_summary import summarize


class EffortTests(unittest.TestCase):
    def make(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        handle.write(text); handle.close(); self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def csv(self, outcomes="10,20,30,40"):
        return "idX,a,b,Effort-,Kloc+,Defects-\n" + "\n".join(
            f"{i},{a},{b},{y},{i},1" for i, (a, b, y) in enumerate(zip(["l", "h", "l", "z"], ["x", "x", "y", "y"], outcomes.split(",")), 1))

    def test_excludes_metadata_and_outcomes_and_keeps_step1(self):
        self.assertEqual(predictor_names(["idX", "a", "Kloc+", "Effort-", "Defects-"]), ["a"])
        self.assertEqual(summarize(self.make("a,Effort-\nl,1\n"))["effort"], "Effort-")

    def test_group_split_reproducible_and_missing_unseen_proposal(self):
        result = evaluate(self.make(self.csv()))
        again = evaluate(self.make(self.csv()))
        self.assertTrue(set(result["train_groups"]).isdisjoint(result["test_groups"]))
        self.assertEqual(result["predictions"], again["predictions"])
        proposal = {"a": None, "b": "new"}
        value = predict_proposal(result["model"], proposal, result["predictor_names"])
        self.assertTrue(np.isfinite(value))

    def test_identical_descriptors_cannot_cross_boundary(self):
        text = "idX,a,b,Effort-,Kloc+,Defects-\n" + "\n".join(
            f"{i},{a},{b},{10 + i},1,1" for i, (a, b) in enumerate(
                [("l", "x"), ("l", "x"), ("h", "x"), ("h", "x"), ("z", "y"), ("z", "y")], 1))
        result = evaluate(self.make(text), test_size=0.34)
        descriptor = lambda row: (str(row["a"]), str(row["b"]))
        train_configs = {descriptor(row) for row in result["train_rows"]}
        test_configs = {descriptor(row) for row in result["test_rows"]}
        self.assertTrue(train_configs.isdisjoint(test_configs))
        self.assertEqual(len(result["train_rows"]) + len(result["test_rows"]), 6)

    def test_actual_missing_training_value_and_target_filter(self):
        path = self.make("idX,a,b,Effort-,Kloc+,Defects-\n1,l,x,10,1,1\n2,,y,20,2,1\n3,h,x,NaN,3,1\n4,h,y,40,4,1\n")
        result = evaluate(path)
        self.assertEqual(result["train_count"] + result["test_count"], 3)
        self.assertTrue(np.isfinite(result["mae"]))

    def test_held_out_outcomes_do_not_change_training_or_baseline(self):
        first = evaluate(self.make(self.csv("10,20,30,40")))
        test_configurations = set(first["test_configurations"])
        changed = ["999" if "\x1f".join((a, b)) in test_configurations else value
                   for (a, b, value) in zip(["l", "h", "l", "z"], ["x", "x", "y", "y"], ["10", "20", "30", "40"])]
        second = evaluate(self.make(self.csv(",".join(changed))))
        self.assertEqual(first["train_count"], second["train_count"])
        self.assertEqual(first["training_mean"], second["training_mean"])
        self.assertEqual(first["predictions"], second["predictions"])


if __name__ == "__main__":
    unittest.main()
