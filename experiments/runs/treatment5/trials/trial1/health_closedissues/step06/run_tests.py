import contextlib, io, sys, unittest
from pathlib import Path
root=Path(__file__).parent
sys.path.insert(0,str(root/'tests'))
suite=unittest.defaultTestLoader.discover(str(root/'tests'))
with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
    result=unittest.TextTestRunner(verbosity=2).run(suite)
print(out.getvalue(),end=''); print(err.getvalue(),end='',file=sys.stderr)
print(f"tests_run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}")
raise SystemExit(0 if result.wasSuccessful() else 1)
