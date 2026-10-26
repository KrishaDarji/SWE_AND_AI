import pathlib, sys, tempfile, unittest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import certify

class CertifyTests(unittest.TestCase):
    def test_read_groups_strips_values(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            f.write("ruse,Effort-\n n,1\n l,3\n n,2\n")
            name = f.name
        try:
            self.assertEqual(certify.read_groups(name, "ruse", "n", "l", "Effort-"), {"n": [1.0, 2.0], "l": [3.0]})
        finally:
            pathlib.Path(name).unlink()

    def test_clear_difference_is_certified(self):
        result = certify.certify([1, 2, 1, 2], [10, 11, 9, 10], 1000)
        self.assertLess(result[0], 0)
        self.assertLess(result[1], .05)
        self.assertLess(result[3], 0)

if __name__ == "__main__": unittest.main()
