import copy, unittest
from source.arrival_estimator import DESCRIPTORS, assess

EXPECTED = ["prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced"]

def row(effort, **changes):
    data = {name: "h" for name in EXPECTED}; data.update(changes); data.update({"idX": "metadata", "centerX": "metadata", "YearX": "metadata", "Kloc+": "100", "Defects-": "200", "MonthsX": "9", "Effort-": effort})
    return data

class GuardedRow(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.outcome_accessed = False; self.prediction_done = False; self.descriptor_reads = 0
    def get(self, key, default=None):
        if key == "Effort-":
            self.outcome_accessed = True
            if not self.prediction_done:
                raise AssertionError("current outcome accessed before prediction")
        elif key in EXPECTED:
            self.descriptor_reads += 1
            if self.descriptor_reads >= len(EXPECTED):
                self.prediction_done = True
        return super().get(key, default)

class FlowTests(unittest.TestCase):
    def test_exact_22_descriptors(self):
        self.assertEqual(DESCRIPTORS, EXPECTED)
        self.assertEqual(len(DESCRIPTORS), 22)
        self.assertFalse(any(name.endswith("X") for name in DESCRIPTORS))
        self.assertNotIn("Effort-", DESCRIPTORS); self.assertNotIn("Kloc+", DESCRIPTORS); self.assertNotIn("Defects-", DESCRIPTORS)

    def test_warmup_invalid_and_independent_cumulative_maes(self):
        got = assess([(str(i), row(e, idX=str(i))) for i, e in enumerate([10, 20, "nan", 40, 60], 1)])
        self.assertEqual([x["status"] for x in got], ["warm-up", "warm-up", "invalid-effort", "scored", "scored"])
        self.assertEqual(got[3]["prior_training_count"], 2); self.assertEqual(got[3]["prediction"], 15.0)
        self.assertEqual(got[3]["cumulative_mae"], 25.0); self.assertEqual(got[3]["baseline_cumulative_mae"], 25.0)
        self.assertAlmostEqual(got[4]["cumulative_mae"], 30.833333333333336)
        self.assertAlmostEqual(got[4]["baseline_cumulative_mae"], 30.833333333333336)

    def test_prediction_precedes_current_outcome(self):
        guarded = [("a", row(10)), ("b", row(20)), ("c", GuardedRow(row(30)))]
        got = assess(guarded)
        self.assertEqual(got[2]["prediction"], 15.0); self.assertTrue(guarded[2][1].outcome_accessed)

    def test_metadata_and_non_target_outcomes_do_not_change_prediction(self):
        base = [("a", row(10)), ("b", row(20)), ("c", row(30, idX="c"))]
        changed = copy.deepcopy(base); changed[0][1].update(idX="changed", centerX="changed", YearX="changed", **{"Kloc+": "999", "Defects-": "999", "MonthsX": "999"})
        self.assertEqual(assess(base)[2]["prediction"], assess(changed)[2]["prediction"])

    def test_current_and_future_outcomes_do_not_change_frozen_prediction(self):
        first = [("a", row(10)), ("b", row(20)), ("c", row(30)), ("d", row(40))]
        current_changed = copy.deepcopy(first); current_changed[2][1]["Effort-"] = 999
        future_changed = copy.deepcopy(first); future_changed[3][1]["Effort-"] = 999
        self.assertEqual(assess(first)[2]["prediction"], assess(current_changed)[2]["prediction"])
        self.assertEqual(assess(first)[2], assess(future_changed)[2])

    def test_earlier_snapshots_remain_unchanged(self):
        rows = [("a", row(10)), ("b", row(20)), ("c", row(30)), ("d", row(40))]
        got = assess(rows); snapshot = copy.deepcopy(got[:3]); got[3]["prediction"] = -1
        self.assertEqual(got[:3], snapshot)

if __name__ == "__main__": unittest.main()
