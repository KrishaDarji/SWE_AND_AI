import io
import json
import math
import pathlib
import sys
import unittest

SOURCE = pathlib.Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from nasa_summary import read_rows, summarize


class SummaryTests(unittest.TestCase):
    def test_numeric_and_categorical_known_results(self):
        result = summarize(io.StringIO("idX,kind,Kloc+,Effort-,Defects-\n1,a,2,4,6\n2,a,4,8,10\n"))
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["columns"][1]["frequencies"], {"a": 2})
        self.assertEqual(result["columns"][2]["statistics"]["mean"], 3)
        self.assertAlmostEqual(result["columns"][2]["statistics"]["sd"], math.sqrt(2))

    def test_whitespace_missing_and_nonfinite(self):
        result = summarize(io.StringIO("a,b\n 1 , x \n ? , NaN\n, inf\n"))
        self.assertEqual(result["columns"][0]["missing"], 2)
        self.assertEqual(result["columns"][0]["statistics"]["mean"], 1)
        self.assertEqual(result["columns"][1]["missing"], 2)

    def test_invalid_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize(io.StringIO("Kloc+,Effort-\nnot-a-number,2\n"))

    def test_roles_and_directions(self):
        result = summarize(io.StringIO("idX,YearX,team,Kloc+,Effort-,Defects-\n1,1980,h,2,3,4\n"))
        roles = {item["name"]: item for item in result["roles"]}
        self.assertIn("metadata", roles["idX"]["role"])
        self.assertEqual(roles["Kloc+"]["optimization"], "maximize")
        self.assertEqual(roles["Effort-"]["optimization"], "minimize")

    def test_empty_and_header_only(self):
        self.assertEqual(summarize(io.StringIO(""))["record_count"], 0)
        result = summarize(io.StringIO("a,bX\n"))
        self.assertEqual(result["column_count"], 2)
        self.assertEqual(result["columns"][0]["missing"], 0)


if __name__ == "__main__":
    unittest.main()
