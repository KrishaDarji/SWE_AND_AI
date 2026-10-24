import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import certify

class CertifyTests(unittest.TestCase):
  def test_strong_accuracy_improvement_is_meaningful(self):
    result = certify([.70, .71, .69, .70, .72], [.80, .81, .79, .82, .80])
    assert result["mean_improvement"] > 0
    assert result["meaningful"] == 1.0
    assert result["p_value"] < .05

  def test_error_improvement_is_sign_reversed(self):
    result = certify([.30, .31, .29, .30], [.20, .21, .19, .20], False)
    assert result["mean_improvement"] > 0

  def test_noise_is_not_meaningful(self):
    result = certify([1, 2, 3, 4, 5], [1.01, 1.99, 3.01, 3.99, 5.01])
    assert result["meaningful"] == 0.0

  def test_bad_input_rejected(self):
    try:
        certify([1], [2])
    except ValueError:
        pass
    else:
        raise AssertionError("short samples should fail")

if __name__ == "__main__":
    unittest.main()
