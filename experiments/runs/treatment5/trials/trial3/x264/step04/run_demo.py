import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from flow_evaluator import main

main(sys.argv[1])
