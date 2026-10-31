import runpy, sys
from pathlib import Path

root = Path(__file__).parent
sys.argv = [str(root / "source" / "certify.py"), "/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv", "--threshold", "11.76"]
runpy.run_path(sys.argv[0], run_name="__main__")
