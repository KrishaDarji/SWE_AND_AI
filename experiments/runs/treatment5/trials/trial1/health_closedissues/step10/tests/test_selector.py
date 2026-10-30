import csv
import math
import os
import tempfile
import unittest
from source.selector import compare, load_records, select, summarize

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def row(mre="0.2", acc="0.3", pred="0.4", n="10", criterion="squared_error", leaves="2", impurity="0.1", depth="4"):
    return [mre, acc, pred, n, criterion, leaves, impurity, depth]


def fixture(rows):
    handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
    csv.writer(handle).writerows([FIELDS] + rows)
    handle.close()
    return handle.name


class SelectorTests(unittest.TestCase):
    def test_integration_validity_cap_ids_and_pool(self):
        path = fixture([row(n="101"), row(mre="0.1", n="50"), row(criterion="?"), row(n="bad"), row(mre="nan"),
                        ["0.2", "0.3", "0.4", "10", "squared_error", "2"]])
        try:
            records, counts = load_records(path)
        finally:
            os.unlink(path)
        self.assertEqual(counts, {"raw_total": 6, "valid": 2, "feasible": 1})
        self.assertEqual([record["original_id"] for record in records], [2])
        self.assertEqual(select(records)["original_id"], 2)
        self.assertTrue(all(item["selector"]["original_id"] == 2 and item["baseline"]["original_id"] == 2 for item in compare(records)))

    def test_order_and_all_tie_breakers(self):
        path = fixture([row("1", ".5", ".5"), row("1", ".6", ".5"), row("1", ".6", ".7"), row(".9", "0", "0")])
        try:
            records, _ = load_records(path)
        finally:
            os.unlink(path)
        self.assertEqual(select(records)["original_id"], 4)
        tied = [{"outcomes": {"MRE-": 1, "ACC+": 1, "PRED40+": 1}, "original_id": 2},
                {"outcomes": {"MRE-": 1, "ACC+": 1, "PRED40+": 1}, "original_id": 1}]
        self.assertEqual(select(tied)["original_id"], 1)

    def test_seed_reproducibility_independent_rng_and_arithmetic(self):
        records = [{"original_id": i, "outcomes": {"MRE-": v, "ACC+": .1*i, "PRED40+": .2*i}, "configuration": {}}
                   for i, v in enumerate([.1, .2, .3], 1)]
        first, second = compare(records, [42, 43]), compare(records, [42, 43])
        self.assertEqual([x["baseline"]["original_id"] for x in first], [x["baseline"]["original_id"] for x in second])
        summary = summarize([{"difference": x, "baseline": {"outcomes": {"ACC+": 0, "PRED40+": 0}},
                             "selector": {"outcomes": {"ACC+": 0, "PRED40+": 0}}} for x in [.2, 0, -.1]])
        self.assertEqual((summary["wins"], summary["ties"], summary["losses"]), (1, 1, 1))
        self.assertAlmostEqual(summary["mean_difference"], 1/30)

    def test_invalid_caps_missing_and_empty(self):
        with self.assertRaises(ValueError): load_records("/dev/null", math.inf)
        with self.assertRaises(ValueError): select([])
        with self.assertRaises(ValueError): summarize([], -1)


if __name__ == "__main__":
    unittest.main()
