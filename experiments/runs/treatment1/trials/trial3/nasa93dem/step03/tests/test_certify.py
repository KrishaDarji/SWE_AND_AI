import os
import tempfile
import unittest

from source.certify import bootstrap_median_difference, certify, load_rows


class CertifyTests(unittest.TestCase):
    def test_bootstrap_is_reproducible(self):
        self.assertEqual(bootstrap_median_difference([1, 2, 3], [5, 6, 7], 100),
                         bootstrap_median_difference([1, 2, 3], [5, 6, 7], 100))

    def test_certifies_clear_difference(self):
        data = "pmat,Effort-\nn,1\nn,2\nn,3\nl,10\nl,11\nl,12\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            result = certify(path)
            self.assertTrue(result["certified"])
            self.assertLess(result["median_difference"], 0)
        finally:
            os.unlink(path)

    def test_missing_group_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("pmat,Effort-\nn,1\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                certify(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
