import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from source.nasa93_summary import justify_estimate

DATA = Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class JustificationTests(unittest.TestCase):
    def test_labels_inputs_and_training_only_evidence(self):
        result = justify_estimate(DATA)
        evidence = result["evidence"]
        self.assertEqual(result["proposal"]["descriptor_count"], 22)
        self.assertIn("idX", result["proposal"]["excluded_inputs"])
        self.assertIn("Effort-", result["proposal"]["excluded_inputs"])
        self.assertEqual(evidence["estimate"]["label"], "MODEL_ESTIMATE")
        self.assertEqual(evidence["training_history_reference"]["label"], "TRAINING_HISTORY_REFERENCE")
        self.assertEqual(evidence["historical_associations"]["label"], "HISTORICAL_ASSOCIATION")
        self.assertEqual(evidence["audit"]["label"], "MEASURED_AUDIT_VALUE")
        self.assertEqual(evidence["training_history_reference"]["training_count"], 68)
        self.assertTrue(np.isfinite(evidence["estimate"]["effort"]))

    def test_held_out_outcome_does_not_change_pre_audit_result(self):
        original = justify_estimate(DATA)
        with open(DATA, newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        effort = rows[0].index("Effort-")
        held = original["proposal"]["held_out_row_index"] + 1
        rows[held][effort] = "999999"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as handle:
            csv.writer(handle).writerows(rows)
            altered_path = Path(handle.name)
        self.addCleanup(lambda: altered_path.unlink(missing_ok=True))
        altered = justify_estimate(altered_path)
        self.assertEqual(original["evidence"]["estimate"]["effort"], altered["evidence"]["estimate"]["effort"])
        self.assertEqual(original["evidence"]["training_history_reference"]["effort"], altered["evidence"]["training_history_reference"]["effort"])
        self.assertEqual(original["evidence"]["historical_associations"], altered["evidence"]["historical_associations"])


if __name__ == "__main__":
    unittest.main()
