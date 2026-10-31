import unittest
suite=unittest.defaultTestLoader.discover('tests'); r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())
