import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))
from summary import main
main(sys.argv[1:])
