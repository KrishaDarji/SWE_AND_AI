import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import certify_x264


class CertifyTests(unittest.TestCase):
    def test_decile_and_direction(self):
        difference, low, rest = certify_x264.mean_difference(list(range(1, 21)))
        self.assertEqual(len(low), 2)
        self.assertEqual(len(rest), 18)
        self.assertGreater(difference, 0)

    def test_reproducible_intervals_and_pvalue(self):
        values = [float(x) for x in range(1, 31)]
        _, low, rest = certify_x264.mean_difference(values)
        self.assertEqual(certify_x264.bootstrap_ci(low, rest, 200), certify_x264.bootstrap_ci(low, rest, 200))
        self.assertEqual(certify_x264.permutation_pvalue(low, rest, 200), certify_x264.permutation_pvalue(low, rest, 200))

    def test_csv_reader(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("Performance-\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
            name = handle.name
        self.assertEqual(certify_x264.read_values(name), list(map(float, range(1, 11))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
