import subprocess, sys
from pathlib import Path
root = Path(__file__).parent
cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-p", "test_*.py"]
raise SystemExit(subprocess.run(cmd, cwd=root).returncode)
