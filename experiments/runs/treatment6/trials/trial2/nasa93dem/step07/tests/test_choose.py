import math, pathlib, tempfile, unittest, sys
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa93_summary as ns

class ChooseTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile("w", delete=False, newline="")
        f.write("idX," + ",".join(ns.CATEGORICAL) + ",Kloc+,Effort-,Defects-\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
        f.close(); return f.name
    def row(self, ident, size, effort, defects=1, tag="a"):
        return [ident] + [tag] * 22 + [size, effort, defects]
    def test_cap_ties_positions_and_complete_configuration(self):
        p = self.write([self.row("dup", 10, 5, 8, "a"), self.row("dup", 10, 5, 7, "b"), self.row("x", 11, 6)])
        r = ns.choose(p, 6); self.assertEqual(r["project"]["position"], 4); self.assertEqual(r["project"]["idX"], "x")
        self.assertEqual(len(r["project"]["descriptors"]), 22); self.assertEqual(r["valid_count"], 3)
        p = self.write([self.row("a", 10, 5), self.row("b", 12, 6), self.row("c", 12, 6)])
        self.assertEqual(ns.choose(p, 6)["project"]["position"], 3)
    def test_exact_inclusion_and_over_budget_exclusion(self):
        p = self.write([self.row("over", 1000, 101), self.row("exact", 9, 100), self.row("low", 8, 1)])
        r = ns.choose(p, 100); self.assertEqual(r["project"]["idX"], "exact"); self.assertEqual(r["over_budget_count"], 1)
    def test_invalid_effort_nonfinite_missing_descriptor_outcomes(self):
        rows = [self.row("negative", 99, -1), self.row("nan", 99, "NaN"), self.row("missing", 99, 1), self.row("badsize", "bad", 1), self.row("baddef", 99, 1, "inf")]
        rows[-1][-1] = "inf"; rows[2][2] = "?"
        p = self.write(rows)
        with self.assertRaisesRegex(ValueError, "no valid"): ns.choose(p, 100)
    def test_invalid_cap_empty_and_no_feasible(self):
        p = self.write([self.row("a", 1, 2)])
        for cap in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError): ns.choose(p, cap)
        with self.assertRaisesRegex(ValueError, "no feasible"): ns.choose(p, 1)
        with self.assertRaisesRegex(ValueError, "no parsed"): ns.choose(self.write([]), 1)
    def test_malformed_rows_and_original_position_counts(self):
        f = tempfile.NamedTemporaryFile("w", delete=False, newline="")
        f.write("idX," + ",".join(ns.CATEGORICAL) + ",Kloc+,Effort-,Defects-\n")
        f.write(",".join(map(str, self.row("bad", 999, 1))) + ",extra\n")
        f.write(",".join(map(str, self.row("winner", 5, 1))) + "\n"); f.close()
        r = ns.choose(f.name, 1); self.assertEqual(r["project"]["position"], 3); self.assertEqual(len(r["invalid_rows"]), 1)

if __name__ == "__main__": unittest.main()
