import tempfile
import unittest
import csv
from pathlib import Path
import numpy as np
from source.nasa93_summary import (build_model, descriptor_columns, evaluate,
    fit_final_model, load_csv, predict_proposal, prepare_learning_data)

DATA = Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")

class EstimationTests(unittest.TestCase):
    def test_exclusions_and_group_split_reproducible(self):
        a = evaluate(DATA); b = evaluate(DATA)
        self.assertEqual(len(a["predictors"]), 22)
        self.assertNotIn("idX", a["predictors"]); self.assertNotIn("Kloc+", a["predictors"])
        self.assertTrue(np.array_equal(a["test_indices"], b["test_indices"]))
        _, _, X, _, groups = prepare_learning_data(DATA)
        for i in a["test_indices"]:
            self.assertFalse(any(groups[i] == groups[j] for j in a["train_indices"]))

    def test_missing_unseen_and_proposal(self):
        a = evaluate(DATA); proposal = {c: None for c in a["predictors"]}
        self.assertTrue(np.isfinite(predict_proposal(a["model"], a["predictors"], proposal)))
        proposal[a["predictors"][0]] = "never-seen"
        self.assertTrue(np.isfinite(predict_proposal(a["model"], a["predictors"], proposal)))

    def test_missing_is_explicit_category(self):
        with open(DATA, encoding="utf-8") as source:
            lines = source.readlines()
        header = lines[0].strip().split(",")
        lines = [lines[0]] + [line.replace("h", "", 1) for line in lines[1:]]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.writelines(lines)
            missing_path = Path(handle.name)
        self.addCleanup(lambda: missing_path.unlink(missing_ok=True))
        a = evaluate(missing_path)
        encoder = a["model"].named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"]
        self.assertTrue(any("<MISSING>" in category for categories in encoder.categories_ for category in categories))

    def test_final_model_is_distinct_from_holdout(self):
        a = evaluate(DATA); model, predictors = fit_final_model(DATA)
        self.assertEqual(len(predictors), len(a["predictors"]))
        self.assertIsNot(model, a["model"])

    def test_training_only_preprocessing(self):
        a = evaluate(DATA)
        names = a["model"].named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"].categories_
        _, _, X, _, _ = prepare_learning_data(DATA)
        train_values = [{X[i][c] for i in a["train_indices"]} for c in a["predictors"]]
        for categories, values in zip(names, train_values):
            self.assertTrue(set(categories).issubset(values))

    def test_held_out_outcomes_are_isolated(self):
        original = evaluate(DATA)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as handle:
            with open(DATA, newline="", encoding="utf-8") as source:
                rows = list(csv.reader(source))
            effort = rows[0].index("Effort-")
            for index in original["test_indices"]:
                rows[index + 1][effort] = str(float(rows[index + 1][effort]) + 999999.0)
            csv.writer(handle).writerows(rows)
            altered_path = Path(handle.name)
        self.addCleanup(lambda: altered_path.unlink(missing_ok=True))
        altered = evaluate(altered_path)
        self.assertTrue(np.array_equal(original["train_indices"], altered["train_indices"]))
        np.testing.assert_allclose(original["predictions"], altered["predictions"])
        self.assertEqual(original["training_mean"], altered["training_mean"])

if __name__ == "__main__": unittest.main()
