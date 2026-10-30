import unittest
from pathlib import Path
suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'))
raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
