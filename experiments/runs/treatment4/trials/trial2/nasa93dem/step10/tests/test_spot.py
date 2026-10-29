import tempfile
import unittest
from pathlib import Path

from source.nasa93_summary import spot_inconsistencies


class SpotTests(unittest.TestCase):
    def fixture(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text); handle.close()
        path = Path(handle.name); self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_synthetic_outlier_and_ordinary_record(self):
        result = spot_inconsistencies(self.fixture(
            "idX,Kloc+,Effort-,Defects-\n" +
            "a,10,10,1\nb,11,11,1\nc,9,9,1\nd,10000,10,1\n"))
        self.assertEqual(result["total_flag_count"], 1)
        self.assertEqual(result["flags"][0]["record_identifier"], "d")
        self.assertEqual(result["flags"][0]["field"], "Kloc+")

    def test_missing_nonfinite_negative_and_constant(self):
        result = spot_inconsistencies(self.fixture(
            "idX,Kloc+,Effort-,Defects-\na,1,1,2\nb,,nan,2\nc,-1,1,2\n"))
        self.assertTrue(any(f["reason"] == "negative value" for f in result["flags"]))
        self.assertEqual(result["fields"][2]["robust_scale"], 0.0)
        self.assertEqual(result["fields"][0]["valid_count"], 1)

    def test_small_empty_and_threshold_validation(self):
        result = spot_inconsistencies(self.fixture("idX,Kloc+\na,1\n"))
        self.assertEqual(result["total_flag_count"], 0)
        with self.assertRaises(ValueError):
            spot_inconsistencies(self.fixture("idX,Kloc+\na,1\n"), -1)


if __name__ == "__main__": unittest.main()
