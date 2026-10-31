import unittest
from tests.test_repair import RepairTests
suite = unittest.defaultTestLoader.loadTestsFromTestCase(RepairTests)
raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
