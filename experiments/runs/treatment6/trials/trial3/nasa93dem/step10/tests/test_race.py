import math, pathlib, sys, unittest
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa_summary as n

class TestRace(unittest.TestCase):
    def row(self, pos, effort, token, ident="dup"):
        return {"position": pos, "values": {"idX": ident, "Effort-": effort,
                **{c: token for c in n.PREDICTORS}}}

    def test_seed_results_are_grouped_reproducible_and_keep_identity(self):
        rows = [self.row(i, str(i), "a" if i < 3 else "b", "same") for i in range(2, 8)]
        a = n.race(rows, seeds=[42], tolerance=0)
        b = n.race(rows, seeds=[42], tolerance=0)
        self.assertEqual(a, b)
        result = a["seeds"][0]
        self.assertTrue(set(x["position"] for x in result["scored"]).issubset(set(range(2, 8))))
        self.assertTrue(all(x["idX"] == "same" for x in result["scored"]))
        self.assertEqual(result["train_groups"] + result["test_groups"], 2)

    def test_training_only_mean_and_outcome_mutation(self):
        rows = [self.row(i, str(i), "a" if i < 4 else "b") for i in range(2, 8)]
        first = n.race(rows, seeds=[42])["seeds"][0]
        _, heldout = n.split_groups(rows, 42)
        for p in heldout: p["values"]["Effort-"] = "999999"
        second = n.race(rows, seeds=[42])["seeds"][0]
        self.assertEqual(first["train_rows"], second["train_rows"])
        self.assertEqual([(x["prediction"], x["baseline_prediction"]) for x in first["scored"]],
                         [(x["prediction"], x["baseline_prediction"]) for x in second["scored"]])

    def test_exclusions_zero_and_no_evaluable(self):
        rows = [self.row(1, "bad", "a"), self.row(2, "0", "b"), self.row(3, "NaN", "c"), self.row(4, "2", "d")]
        out = n.race(rows, seeds=[42])["seeds"][0]
        self.assertGreaterEqual(out["excluded_train_targets"] + out["excluded_test_targets"], 1)
        self.assertEqual([x["actual"] for x in out["scored"]].count(0.0) <= 1, True)

    def test_classification_and_tolerance_validation(self):
        self.assertEqual(n.classify_difference(1, .1), "win")
        self.assertEqual(n.classify_difference(-1, .1), "loss")
        self.assertEqual(n.classify_difference(.1, .1), "tie")
        self.assertEqual(n.classify_difference(float("nan"), .1), "undefined")
        with self.assertRaises(ValueError): n.race([], tolerance=-1)

    def test_insufficient_groups_and_summary_counts(self):
        out = n.race([self.row(1, "1", "a")], seeds=[42])
        self.assertEqual(out["seeds"][0]["status"], "insufficient groups")
        rows = [self.row(i, str(i), "a" if i < 4 else "b") for i in range(2, 8)]
        out = n.race(rows, seeds=[42, 43], tolerance=1e9)
        self.assertEqual(out["comparable_count"], 2)
        self.assertEqual(out["ties"], 2)

if __name__ == "__main__": unittest.main()
