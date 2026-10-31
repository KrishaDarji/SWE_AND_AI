import subprocess, sys
from pathlib import Path
p = Path(__file__).parent
with open(p / "test_output_attempt01.txt", "w") as out:
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(p / "tests"), "-p", """test_*.py"""], cwd=p, stdout=out, stderr=subprocess.STDOUT, check=False)
