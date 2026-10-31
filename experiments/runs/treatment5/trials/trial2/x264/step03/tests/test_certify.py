import math
import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import certify


class CertifyTests(unittest.TestCase):
    def test_direction_and_interval(self):
        r = certify.welch([10, 11, 12], [8, 9, 10])
        self.assertGreater(r["difference"], 0)
        self.assertAlmostEqual(r["difference"], 2.0)
        self.assertLess(r["ci"][0], r["difference"])

    def test_unequal_variance(self):
        r = certify.welch([1, 2, 3], [10, 20, 30, 40])
        expected = (1/3 + (500/3)/4) ** 2 / ((1/3)**2/2 + ((500/3)/4)**2/3)
        self.assertAlmostEqual(r["df"], expected, places=10)

    def test_missing_and_insufficient(self):
        self.assertEqual(certify.load_groups(os.path.join(os.path.dirname(__file__), "fixture.csv"))[2], 1)
        with self.assertRaises(ValueError): certify.welch([1], [2, 3])
        with self.assertRaises(ValueError): certify.welch([1, math.inf], [2, 3])

    def test_zero_variance(self):
        equal = certify.welch([2, 2], [2, 2])
        unequal = certify.welch([1, 1], [2, 2])
        self.assertEqual(equal["status"], "undefined")
        self.assertEqual(unequal["status"], "undefined")
        self.assertNotIn("p", equal)
        self.assertNotIn("ci", unequal)

    def test_one_constant_one_varying(self):
        r = certify.welch([2, 2, 2], [1, 2, 3])
        self.assertEqual(r["status"], "valid")
        self.assertTrue(math.isfinite(r["p"]))
        self.assertEqual(len(r["ci"]), 2)

    def test_cli_undefined_inference(self):
        import contextlib
        import io
        path = os.path.join(os.path.dirname(__file__), "constant_fixture.csv")
        output = io.StringIO()
        with contextlib.redirect_stdout(output): certify.main(["certify.py", path])
        text = output.getvalue()
        self.assertIn("status=undefined", text)
        self.assertIn("no Welch p-value or interval", text)
        self.assertNotIn("two-sided p=", text)

    def test_threshold_uncertain(self):
        r = certify.welch([0, 2, 4, 6], [-2, 0, 2, 4], threshold=1.0)
        self.assertTrue(r["exceeds"])
        self.assertFalse(r["beyond"])
        with self.assertRaises(ValueError): certify.welch([1, 2], [1, 2], threshold=-1)

    def test_known_t_values_small_large(self):
        self.assertAlmostEqual(certify.student_t_cdf(0, 1), .5, places=14)
        self.assertAlmostEqual(certify.student_t_cdf(1, 1), .75, places=12)
        self.assertAlmostEqual(2 * (1 - certify.student_t_cdf(2.22813885, 10)), .05, places=6)
        self.assertAlmostEqual(certify.student_t_cdf(8, 1e8), .9999999999999994, places=14)


if __name__ == "__main__": unittest.main()
