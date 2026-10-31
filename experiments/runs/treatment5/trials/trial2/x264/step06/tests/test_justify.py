import importlib.util
import unittest

SPEC = importlib.util.spec_from_file_location("justify", "source/justify.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class JustifyTests(unittest.TestCase):
    def rows(self, values):
        return [{"a": str(i), "b": "x", "Performance-": v} for i, v in enumerate(values)]

    def test_selection_and_tie_break(self):
        out = MOD.analyze(self.rows(["10", "3", "3"]))
        self.assertEqual(out["record_id"], 2)
        self.assertEqual(out["configuration"]["a"], "1")

    def test_reference_arithmetic(self):
        out = MOD.analyze(self.rows(["10", "4"]))
        self.assertEqual(out["absolute_difference"], 6)
        self.assertEqual(out["percentage_improvement"], 60)

    def test_complete_config_and_labels(self):
        out = MOD.analyze(self.rows(["5"]))
        text = MOD.render(out)
        self.assertIn("a=0", text)
        self.assertIn("Measured", text)
        self.assertIn("Historical reference", text)

    def test_missing_nonfinite_and_zero_reference(self):
        self.assertEqual(MOD.analyze(self.rows(["nan", "2"]))["record_id"], 2)
        out = MOD.analyze(self.rows(["0", "-1"]))
        self.assertEqual(out["absolute_difference"], 1)
        self.assertIsNone(out["percentage_improvement"])
        text = MOD.render(out)
        self.assertIn("Absolute difference (reference - recommendation): 1.", text)
        self.assertIn("percentage improvement unavailable: reference is zero", text)

    def test_no_usable_records(self):
        out = MOD.analyze(self.rows(["?", "inf"]))
        self.assertEqual(out["status"], "no usable records")


if __name__ == "__main__":
    unittest.main()
