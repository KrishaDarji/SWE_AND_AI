import tempfile
import unittest
from pathlib import Path

from source.nasa93_summary import choose_settings


HEADER = "idX,metaX,setting,Kloc+,Effort-,Defects-\n"


class ChooseTests(unittest.TestCase):
    def fixture(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Path(handle.name)

    def test_budget_boundary_and_maximum(self):
        result = choose_settings(self.fixture(HEADER + "a,x,low,50,100,1\nb,x,high,80,101,1\n"), 100)
        self.assertEqual(result["feasible_count"], 1)
        self.assertEqual(result["selected"]["record_identifier"], "a")
        self.assertEqual(result["selected"]["descriptors"], {"setting": "low"})
        self.assertNotIn("metaX", result["selected"]["descriptors"])

    def test_ties_lower_effort_then_earliest(self):
        text = HEADER + "first,x,a,80,50,1\nsecond,x,b,80,40,1\nthird,x,c,80,40,1\n"
        result = choose_settings(self.fixture(text), 50)
        self.assertEqual(result["selected"]["record_identifier"], "second")

    def test_invalid_budget(self):
        with self.assertRaises(ValueError): choose_settings(self.fixture(HEADER), -1)

    def test_missing_outcome_and_no_feasible(self):
        with self.assertRaises(ValueError): choose_settings(self.fixture("idX,setting,Effort-\na,x,1\n"))
        result = choose_settings(self.fixture(HEADER + "a,x,a,80,51,1\nb,x,b,,10,1\n"), 50)
        self.assertEqual(result["status"], "no feasible records")
        self.assertEqual(result["feasible_count"], 0)


if __name__ == "__main__":
    unittest.main()
