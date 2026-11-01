import subprocess, sys
from pathlib import Path
root = Path(__file__).parent
dataset = Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
raise SystemExit(subprocess.run([sys.executable, str(root / "source" / "nasa93_summary.py"), str(dataset)], cwd=root).returncode)
